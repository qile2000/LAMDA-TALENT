from typing import List, Optional, Tuple, Callable, Dict, Any

import numpy as np
import torch

from TALENT.model.lib.realmlp.data.data import DictDataset, ParallelDictDataLoader, TaskType, ValDictDataLoader
from TALENT.model.lib.realmlp.nn_models.base import set_hp_context, SequentialLayer, Layer, Variable
from TALENT.model.lib.realmlp.nn_models.models import NNFactory
from TALENT.model.lib.realmlp.training.lightning_callbacks import StopAtEpochsCallback, HyperparamCallback, L1L2RegCallback, \
    ModelCheckpointCallback
from TALENT.model.lib.realmlp.training.coord import HyperparamManager
from TALENT.model.lib.realmlp.training.logging import Logger
from TALENT.model.lib.realmlp.training.metrics import Metrics, mse, cross_entropy
from TALENT.model.lib.realmlp.alg_interfaces.base import SplitIdxs, InterfaceResources


class NNCreator:
    def __init__(self, fit_params: Optional[List[Dict[str, Any]]] = None, **config):
        self.fit_params = fit_params
        self.config = config
        self.device_info = None  # todo: allow better configurability, including mps?
        self.n_tt_splits = None
        self.n_tv_splits = None
        self.static_model = None

        self.factory = self.config.get('factory', None)
        if self.factory is None:
            self.factory = NNFactory(**self.config)

        self.hp_manager = HyperparamManager(**self.config)

        # Data Info
        self.is_cv = None
        self.train_idxs = None
        self.val_idxs = None
        self.n_classes = None

    def setup_from_dataset(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources):
        torch.backends.cuda.matmul.allow_tf32 = False  # todo: should we do this?
        # todo: allow preprocessing on CPU and then only put batches on GPU in data loader?
        gpu_devices = interface_resources.gpu_devices
        self.device_info = gpu_devices[0] if len(gpu_devices) > 0 else 'cpu'

        # the code below requires all splits to have the same number of sub-splits
        assert np.all([idxs_list[i].train_idxs.shape[0] == idxs_list[0].train_idxs.shape[0]
                       for i in range(len(idxs_list))])

        # we can then decompose the overall number of sub-splits into the number of splits
        # and the number of sub-splits per split
        self.n_tt_splits = len(idxs_list)
        self.n_tv_splits = idxs_list[0].train_idxs.shape[0]

        self.is_cv = idxs_list[0].val_idxs is not None
        assert np.all([(split_idxs.val_idxs is not None) == self.is_cv for split_idxs in idxs_list])

        y_cat_sizes = ds.tensor_infos['y'].get_cat_sizes().numpy()
        self.n_classes = y_cat_sizes[0]
        self.train_idxs = torch.cat([split_idxs.train_idxs for split_idxs in idxs_list], dim=0)
        self.val_idxs = torch.cat([split_idxs.val_idxs for split_idxs in idxs_list], dim=0) if self.is_cv else None

    def get_criterions(self) -> Tuple[Callable, str]:
        task_type = TaskType.REGRESSION if self.n_classes == 0 else TaskType.CLASSIFICATION
        # train criterion
        # todo: add more options?
        train_metric_name = self.config.get('train_metric_name', None)
        if train_metric_name is None:
            train_criterion = mse if self.n_classes == 0 else cross_entropy  # defaults
        elif train_metric_name == 'mse':
            train_criterion = mse
        elif train_metric_name == 'cross_entropy':
            train_criterion = cross_entropy
        else:
            train_criterion = lambda y_pred, y, mn=train_metric_name: Metrics.apply(y_pred, y, mn)
        # else:
        #     raise ValueError(f'{train_metric_name=} is currently not supported')

        val_criterion = self.config.get('val_metric_name', Metrics.default_metric_name(task_type))
        return train_criterion, val_criterion

    def create_model(self, ds: DictDataset, idxs_list: List[SplitIdxs]):
        ds = ds.to(self.device_info)
        # Create static model
        model_fitter = self.factory.create(ds.tensor_infos)
        static_fitter, dynamic_fitter = model_fitter.split_off_dynamic()
        self.static_model, ds = static_fitter.fit_transform(ds)

        # in the single split case, we can already apply static fitters to the dataset
        is_single_split = len(idxs_list) == 1 and idxs_list[0].n_trainval_splits == 1

        models = []
        # Build non-static models
        for split_idx, split_idxs in enumerate(idxs_list):
            # fit initial values only on train
            model_idx = 0
            with torch.no_grad():
                # fit initial values on train_ds
                for sub_idx in range(split_idxs.n_trainval_splits):
                    if 'feature_importances' in self.config:
                        self.hp_manager.get_more_info_dict()['feature_importances'] = \
                            self.config['feature_importances'][model_idx]
                    if 'fixed_weight' in self.config:
                        self.hp_manager.get_more_info_dict()['fixed_weight'] = \
                            self.config['fixed_weight'][model_idx]
                    train_ds = ds.get_sub_dataset(split_idxs.train_idxs[sub_idx, :])
                    # still call it 'trainval_ds'
                    # because that's what the clipping and output standardization layers use
                    self.hp_manager.get_more_info_dict()['trainval_ds'] = train_ds
                    data_fitter, individual_fitter = dynamic_fitter.split_off_individual()
                    ram_limit_gb = self.config.get('init_ram_limit_gb', 1.0)
                    with set_hp_context(self.hp_manager):
                        torch.manual_seed(split_idxs.split_seed)  # should not be necessary, but just in case
                        data_tfm, tfmd_ds = data_fitter.fit_transform_subsample(
                            train_ds, ram_limit_gb, needs_tensors=individual_fitter.needs_tensors)

                    torch.manual_seed(split_idxs.sub_split_seeds[sub_idx])
                    with set_hp_context(self.hp_manager):
                        individual_tfm = individual_fitter.fit_transform_subsample(
                            tfmd_ds, ram_limit_gb=ram_limit_gb, needs_tensors=False)[0]
                    if is_single_split and self.config.get('allow_single_split_opt', True):
                        self.static_model = SequentialLayer([self.static_model, data_tfm])
                        models.append(individual_tfm)
                    else:
                        models.append(SequentialLayer([data_tfm, individual_tfm]))
                    self.hp_manager.get_more_info_dict()['trainval_ds'] = None

                    model_idx += 1

        # print(f'{models[0]=}')
        # for p in models[0].parameters():
        #     print(str(p.context.scope))
        vectorized_model = models[0].stack(models).to(self.device_info)

        fixed_init_params: Optional[List[Variable]] = self.config.get('fixed_init_params', None)
        if fixed_init_params is not None:
            fixed_init_param_patterns = self.config['fixed_init_param_patterns']
            reinit_lr_factor = self.config.get('reinit_lr_factor', 1.0)
            for param, fixed_init_param in zip(vectorized_model.parameters(), fixed_init_params):
                scope_str = str(param.context.scope)
                # print(scope_str)
                if any(pattern in scope_str for pattern in fixed_init_param_patterns):
                    print(f'Initializing {scope_str} from fixed parameters')
                    with torch.no_grad():
                        param.copy_(fixed_init_param)
                        param: Variable = param
                        param.hyper_factors['lr'] = reinit_lr_factor * fixed_init_param.hyper_factors.get('lr', 1.0)
                        # param.hyper_factors['wd'] = 0.0
                else:
                    print(f'Initializing {scope_str} newly')

        return vectorized_model

    def create_callbacks(self, model: Layer, logger: Logger):
        callbacks = [HyperparamCallback(self.hp_manager), L1L2RegCallback(self.hp_manager, model)]
        # if validation
        if self.is_cv and self.fit_params is None and self.config.get('use_best_epoch', True):
            callbacks.append(ModelCheckpointCallback(self.n_tt_splits, n_tv_splits=self.n_tv_splits,
                                                     use_best_mean_epoch=self.config.get('use_best_mean_epoch_for_cv',
                                                                                         False),
                                                     restore_best=self.config.get('use_best_epoch', True)))
        elif self.fit_params is not None:
            if self.config.get('use_best_mean_epoch_for_refit', True):
                stop_epochs = [[params['stop_epoch']] * self.n_tv_splits for params in self.fit_params]
            else:
                if 'best_indiv_stop_epochs' not in self.fit_params[0] \
                        or len(self.fit_params[0]['best_indiv_stop_epochs']) != self.n_tv_splits:
                    raise ValueError(f'Setting use_best_mean_epoch_for_refit=False '
                                     f'requires setting use_best_epoch=True and n_cv==n_refit')
                stop_epochs = [params['best_indiv_stop_epochs'] for params in self.fit_params]
            callbacks.append(
                StopAtEpochsCallback(stop_epochs=stop_epochs, n_models=self.n_tv_splits, model=model,
                                     logger=logger))
            # only for debugging:
            # callbacks.append(ValidationCallback(ds=ds, val_idxs=test_idxs,
            #                    metric_name=Metrics.default_metric_name(task_type),
            #                    logger=logger, n_models=n_models, n_parallel=n_parallel,
            #                    save_best_params=False,
            #                    val_batch_size=self.config.get('predict_batch_size', 256)))
        return callbacks

    def create_dataloaders(self, ds: DictDataset):
        ds = ds.to(self.device_info)
        ds = self.static_model(ds)
        train_dl = ParallelDictDataLoader(ds, self.train_idxs, batch_size=self.config.get('batch_size', 256),
                                          shuffle=True, drop_last=True, adjust_bs=self.config.get('adjust_bs', False))
        val_dl = None
        if self.is_cv and self.fit_params is None:
            val_dl = ValDictDataLoader(ds, self.val_idxs, val_batch_size=self.config.get('predict_batch_size', 1024))
        return train_dl, val_dl
