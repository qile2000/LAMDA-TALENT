import copy
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import numpy as np
import torch
import pytorch_lightning as pl
import logging

from TALENT.model.lib.realmlp import utils
from TALENT.model.lib.realmlp.data.data import DictDataset
from TALENT.model.lib.realmlp.nn_models.base import Layer, Variable
from TALENT.model.lib.realmlp.nn_models.models import NNFactory
from TALENT.model.lib.realmlp.sklearn.default_params import DefaultParams
from TALENT.model.lib.realmlp.torch_utils import cat_if_necessary
from TALENT.model.lib.realmlp.training.lightning_modules import TabNNModule
from TALENT.model.lib.realmlp.training.logging import Logger
from TALENT.model.lib.realmlp.alg_interfaces.alg_interfaces import AlgInterface
from TALENT.model.lib.realmlp.alg_interfaces.base import SplitIdxs, InterfaceResources, RequiredResources


class NNAlgInterface(AlgInterface):
    def __init__(self, fit_params: Optional[List[Dict[str, Any]]] = None, **config):
        super().__init__(fit_params=fit_params, **config)
        self.model: Optional[TabNNModule] = None
        self.trainer: Optional[pl.Trainer] = None
        self.device = None

    def get_refit_interface(self, n_refit: int, fit_params: Optional[List[Dict]] = None) -> 'AlgInterface':
        return NNAlgInterface(fit_params if fit_params is not None else self.fit_params, **self.config)

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str):
        # the code below requires all splits to have the same number of sub-splits
        assert np.all([idxs_list[i].train_idxs.shape[0] == idxs_list[0].train_idxs.shape[0]
                       for i in range(len(idxs_list))])
        # we can then decompose the overall number of sub-splits into the number of splits
        # and the number of sub-splits per split

        # have the option to change the seeds (for comparing NNs with different random seeds)
        random_seed_offset = self.config.get('random_seed_offset', 0)
        if random_seed_offset != 0:
            idxs_list = [SplitIdxs(train_idxs=idxs.train_idxs, val_idxs=idxs.val_idxs,
                                   test_idxs=idxs.test_idxs, split_seed=idxs.split_seed + random_seed_offset,
                                   sub_split_seeds=[seed + random_seed_offset for seed in idxs.sub_split_seeds],
                                   split_id=idxs.split_id) for idxs in idxs_list]

        # https://stackoverflow.com/questions/74364944/how-to-get-rid-of-info-logging-messages-in-pytorch-lightning
        log = logging.getLogger("pytorch_lightning")
        log.propagate = False
        log.setLevel(logging.ERROR)

        warnings.filterwarnings("ignore", message="You defined a `validation_step` but have no `val_dataloader`.")

        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False  # to be safe wrt rounding errors, but might not be necessary
        # todo: allow preprocessing on CPU and then only put batches on GPU in data loader?
        gpu_devices = interface_resources.gpu_devices
        self.device = gpu_devices[0] if len(gpu_devices) > 0 else 'cpu'
        ds = ds.to(self.device)

        n_epochs = self.config.get('n_epochs', 256)
        self.model = TabNNModule(**utils.join_dicts({'n_epochs': 256, 'logger': logger}, self.config),
                                 fit_params=self.fit_params)
        self.model.compile_model(ds, idxs_list, interface_resources)

        if self.device == 'cpu':
            pl_accelerator = 'cpu'
            pl_devices = 'auto'
        elif self.device == 'mps':
            pl_accelerator = 'mps'
            pl_devices = 'auto'
        elif self.device == 'cuda':
            pl_accelerator = 'gpu'
            pl_devices = [0]
        elif self.device.startswith('cuda:'):
            pl_accelerator = 'gpu'
            pl_devices = [int(self.device[len('cuda:'):])]
        else:
            raise ValueError(f'Unknown device "{self.device}"')

        self.trainer = pl.Trainer(
            accelerator=pl_accelerator,
            devices=pl_devices,
            callbacks=self.model.create_callbacks(),
            max_epochs=n_epochs,
            enable_checkpointing=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            logger=pl.loggers.logger.DummyLogger(),
            enable_model_summary=False,
            log_every_n_steps=1,
        )

        self.trainer.fit(
            model=self.model, train_dataloaders=self.model.train_dl, val_dataloaders=self.model.val_dl
        )

        if hasattr(self.model, 'fit_params'):
            self.fit_params = self.model.fit_params

        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32

        # self.model.to('cpu')  # to allow serialization without GPU issues, but doesn't work

        # print(f'Importances (sorted):', self.get_importances().sort()[0])  # todo

    def predict(self, ds: DictDataset) -> torch.Tensor:
        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model.to(self.device)
        ds = ds.to(self.device)
        ds_x, _ = ds.split_xy()
        y_pred = self.trainer.predict(model=self.model, dataloaders=self.model.get_predict_dataloader(ds_x))
        y_pred = cat_if_necessary(y_pred, dim=-2).to('cpu')  # concat along batch dimension
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
        # self.model.to('cpu')  # to allow serialization without GPU issues, but doesn't work
        return y_pred

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int]) -> RequiredResources:
        tensor_infos = ds.tensor_infos
        factory = self.config.get('factory', None)
        if factory is None:
            factory = NNFactory(**self.config)
        fitter = factory.create(tensor_infos)
        static_fitter, dynamic_fitter = fitter.split_off_dynamic()
        static_tensor_infos = static_fitter.forward_tensor_infos(tensor_infos)
        n_params = fitter.get_n_params(tensor_infos)
        n_forward = fitter.get_n_forward(tensor_infos)
        n_parallel = max(n_cv, n_refit) * n_splits
        batch_size = self.config.get('batch_size', 256)
        n_epochs = self.config.get('n_epochs', 256)
        # per-element RAM usage:
        # continuous data requires 4 bytes for forward pass and 4 for backward pass
        # categorical data requires 8 bytes for forward pass (because torch.long is required) and none for backward pass
        pass_memory = n_forward * batch_size * 8  # initial batch size ignored
        ds_size_gb = ds.n_samples * sum([ti.get_n_features() * (8 if ti.is_cat() else 4)
                                         for ti in static_tensor_infos.values()]) / (1024 ** 3)
        ds_ram_gb = 5 * ds_size_gb
        # ds_ram_gb = 3 * task_info.get_ds_size_gb() / (1024**3)
        param_memory = 5 * n_params * 8  # 5 because of model, model copy, grads, adam mom, adam sq_mom
        fixed_ram_gb = 0.3  # go safe

        # print(f'{pass_memory=}, {param_memory=}')

        # max memory that would be used if the dataset wasn't used
        init_ram_gb_full = n_forward * ds.n_samples * 8
        init_ram_gb_max = 1.5  # todo: rough estimate, a bit larger than what is allowed in fit_transform_subsample()
        init_ram_gb = min(init_ram_gb_max, init_ram_gb_full)
        # init_ram_gb = 1.5

        factor = 1.5  # to go safe on ram
        gpu_ram_gb = fixed_ram_gb + ds_ram_gb + max(init_ram_gb,
                                                    factor * (n_parallel * (pass_memory + param_memory)) / (1024 ** 3))

        gpu_usage = min(1.0, n_parallel / 100)  # rather underestimate it and use up all the ram on the gpu
        # go somewhat safe, should be small anyway
        cpu_ram_gb = 0.3 + ds_ram_gb + 1.3 * (pass_memory + param_memory) / (1024 ** 3)

        time_approx = ds.n_samples * n_epochs * 4e-5 * (2 if n_refit > 0 else 1)
        if self.config.get('use_gpu', True):
            return RequiredResources(time_s=time_approx, n_threads=1.0, cpu_ram_gb=cpu_ram_gb,
                                     n_gpus=1, gpu_usage=gpu_usage, gpu_ram_gb=gpu_ram_gb)
        else:
            return RequiredResources(time_s=time_approx, n_threads=1.0, cpu_ram_gb=cpu_ram_gb + gpu_ram_gb)

    def get_model_ram_gb(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int]):
        tensor_infos = ds.tensor_infos
        factory = self.config.get('factory', None)
        if factory is None:
            factory = NNFactory(**self.config)
        fitter = factory.create(tensor_infos)
        n_params = fitter.get_n_params(tensor_infos)
        n_parallel = max(n_cv, n_refit) * n_splits

        factor = 1.2  # to go safe on ram
        return factor * n_parallel * n_params * 4 / (1024 ** 3)

    def get_importances(self) -> torch.Tensor:
        net: Layer = self.model.model
        params = net.parameters()
        scale = None
        weight = None
        importances_param = self.config.get('feature_importances', None)
        for param in params:
            param: Variable = param
            scope_str = str(param.context.scope)
            if scope_str.endswith('layer-0/scale'):
                scale = param
            elif scope_str.endswith('layer-0/weight'):
                weight = param

            # print(scope_str)

        assert weight is not None

        with torch.no_grad():
            # shape: (vectorized network dims) x n_features
            importances = weight.norm(dim=-1)

            if scale is not None:
                importances *= scale[..., 0, :].abs()

            p = self.config.get('importances_exponent', 1.0)
            importances = importances ** p
            #
            # # hard feature selection
            # n_remove = int(0.9 * importances.shape[-1])
            # new_importances = torch.ones_like(importances)
            # for i in range(importances.shape[0]):
            #     new_importances[i, torch.argsort(importances[i])[:n_remove]] = 0.0
            # importances = new_importances
            # print(importances)

            if importances_param is not None:
                print(f'Using importances_param')
                importances *= importances_param[..., :]

            importances /= (importances.norm(dim=-1, keepdim=True) / np.sqrt(importances.shape[-1]))
            return importances

    def get_first_layer_weights(self, with_scale: bool) -> torch.Tensor:
        net: Layer = self.model.model
        params = net.parameters()
        scale = None
        weight = None
        for param in params:
            param: Variable = param
            scope_str = str(param.context.scope)
            if scope_str.endswith('layer-0/scale'):
                scale = param
            elif scope_str.endswith('layer-0/weight'):
                weight = param
        assert weight is not None
        if scale is not None and with_scale:
            with torch.no_grad():
                return weight * scale[..., 0, :, None]
        else:
            return weight.data

    # todo: have option to move to/from GPU


class RealMLPParamSampler:
    def __init__(self, is_classification: bool):
        self.is_classification = is_classification

    def sample_params(self, seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed=seed)

        hidden_size_options = [[256] * 3, [64] * 5, [512]]

        params = {'num_emb_type': rng.choice(['none', 'pbld', 'pl', 'plr']),
                  'add_front_scale': rng.choice([True, False], p=[0.6, 0.4]),
                  'lr': np.exp(rng.uniform(np.log(2e-2), np.log(3e-1))),
                  'p_drop': rng.choice([0.0, 0.15, 0.3], p=[0.3, 0.5, 0.2]),
                  'wd': rng.choice([0.0, 2e-2]),
                  'plr_sigma': np.exp(rng.uniform(np.log(0.05), np.log(0.5))),
                  'act': rng.choice(['relu', 'selu', 'mish']),
                  'hidden_sizes': hidden_size_options[rng.choice([0, 1, 2], p=[0.6, 0.2, 0.2])]}

        if self.is_classification:
            params['ls_eps'] = rng.choice([0.0, 0.1], p=[0.3, 0.7])

        # print(f'{params=}')

        default_params = DefaultParams.RealMLP_TD_CLASS if self.is_classification else DefaultParams.RealMLP_TD_REG
        return utils.join_dicts(default_params, params)