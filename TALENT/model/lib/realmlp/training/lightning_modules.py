import pytorch_lightning as pl
from typing import List, Optional, Dict, Any
import numpy as np
import torch

from TALENT.model.lib.realmlp.data.data import ParallelDictDataLoader, DictDataset
from TALENT.model.lib.realmlp.alg_interfaces.base import SplitIdxs, InterfaceResources
from TALENT.model.lib.realmlp.nn_models.base import Layer
from TALENT.model.lib.realmlp.optim.optimizers import get_opt_class
from TALENT.model.lib.realmlp.training.nn_creator import NNCreator
from TALENT.model.lib.realmlp.training.logging import StdoutLogger, Logger
from TALENT.model.lib.realmlp.training.metrics import Metrics
from TALENT.model.lib.realmlp.training.scheduling import LearnerProgress


class TabNNModule(pl.LightningModule):
    def __init__(self, n_epochs: int = 256, logger: Optional[Logger] = None,
                 fit_params: Optional[List[Dict[str, Any]]] = None,
                 **config):
        """
        Pytorch Lightning Module for building and training a pytorch NN for tabular data.
        The core of the module is the NNCreatorInterface, which is used to create the model, the callbacks,
        the hyperparameter manager and the dataloaders. The TabNNModule is responsible for the training loop,
        (optional) validation and inference.
        """
        super().__init__()
        self.my_logger = logger or StdoutLogger(verbosity_level=config.get('verbosity', 0))
        # todo: improve this
        self.creator = NNCreator(
            n_epochs=n_epochs, fit_params=fit_params, **config
        )

        self.hp_manager = self.creator.hp_manager
        self.model: Optional[Layer] = None
        self.criterion = None
        self.train_dl = None

        self.progress = LearnerProgress()
        self.progress.max_epochs = n_epochs
        self.fit_params = fit_params

        # Validation
        self.val_preds = []
        self.old_training = None
        self.val_dl = None
        self.save_best_params = True
        self.val_metric_name = None
        self.epoch_mean_val_errors = None
        self.best_mean_val_errors = None
        self.best_mean_val_epochs = None
        self.best_val_errors = None
        self.best_val_epochs = None
        self.has_stopped_list = None

        # LightningModule
        self.automatic_optimization = False

        self.config = config

    def compile_model(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources):
        """
        Method to create the model and all other training dependencies given the dataset and the assigned resources.
        Once this is called, the module is ready for training.
        """
        self.creator.setup_from_dataset(
            ds, idxs_list=idxs_list, interface_resources=interface_resources
        )
        self.model = self.creator.create_model(ds, idxs_list=idxs_list)
        self.train_dl, self.val_dl = self.creator.create_dataloaders(ds)
        self.criterion, self.val_metric_name = self.creator.get_criterions()

    def create_callbacks(self):
        """ Helper method to return callbacks for the trainer.fit callback argument."""
        return self.creator.create_callbacks(self.model, self.my_logger)

    def get_predict_dataloader(self, ds: DictDataset):
        """ Helper method to create a dataloader for inference."""
        ds_x, _ = ds.split_xy()
        ds_x = self.creator.static_model.forward_ds(ds_x)
        idxs_single = torch.arange(ds.n_samples, dtype=torch.long)
        idxs = idxs_single[None, :].expand(
            self.creator.n_tt_splits * self.creator.n_tv_splits, -1
        )

        return ParallelDictDataLoader(ds=ds_x, idxs=idxs,
                                      batch_size=self.creator.config.get("predict_batch_size", 1024))

    # ----- Start LightningModule Methods -----
    def on_fit_start(self):
        self.model.train()
        # mean val errors will not be accurate if all epochs after this yield NaN
        self.best_mean_val_errors = [np.Inf] * self.creator.n_tt_splits
        # epoch 0 counts as before training, epoch 1 is first epoch
        self.best_mean_val_epochs = [0] * self.creator.n_tt_splits
        # don't use simpler notation of the form [[]] * 2 because this will have two references to the same inner array!
        self.best_val_errors = [[np.Inf] * self.creator.n_tv_splits for i in range(self.creator.n_tt_splits)]
        self.best_val_epochs = [[0] * self.creator.n_tv_splits for i in range(self.creator.n_tt_splits)]
        self.has_stopped_list = [[False] * self.creator.n_tv_splits for i in range(self.creator.n_tt_splits)]

    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        opt = self.optimizers()
        # do sum() over models dimension
        loss = self.criterion(output["x_cont"], output["y"]).sum()
        # Callbacks for regularization are called before the backward pass
        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()

        self.progress.total_samples += batch["y"].shape[-2]
        self.progress.epoch_float = (
                self.progress.total_samples / self.train_dl.get_num_iterated_samples()
        )
        return loss

    def on_validation_start(self):
        self.old_training = self.model.training
        self.val_preds = []
        self.model.eval()

    def validation_step(self, batch, batch_idx):
        self.val_preds.append(self.model(batch)["x_cont"])

    def on_validation_epoch_end(self):
        self.model.train(self.old_training)
        self.old_training = None
        y_pred = torch.cat(self.val_preds, dim=-2)
        val_errors = torch.as_tensor(
            [
                Metrics.apply(
                    y_pred[i, :, :], self.val_dl.val_y[i, :, :], self.val_metric_name
                )
                for i in range(y_pred.shape[0])
            ]
        )
        val_errors = val_errors.view(
            self.creator.n_tt_splits, self.creator.n_tv_splits
        )
        mean_val_errors = val_errors.mean(dim=-1)  # mean over cv/refit dimension
        mean_val_error = mean_val_errors.mean().item()

        self.my_logger.log(
            2,
            f"Epoch {self.progress.epoch + 1}/{self.progress.max_epochs}: val error = {mean_val_error:6.6f}",
        )

        use_early_stopping = self.config.get('use_early_stopping', False)
        early_stopping_additive_patience = self.config.get('early_stopping_additive_patience', 20)
        early_stopping_multiplicative_patience = self.config.get('early_stopping_multiplicative_patience', 2)

        current_epoch = self.progress.epoch + 1

        for tt_split_idx in range(self.creator.n_tt_splits):
            use_last_best_epoch = self.config.get('use_last_best_epoch', True)

            has_stopped = self.has_stopped_list[tt_split_idx]

            # compute best single-split validation errors
            for tv_split_idx in range(self.creator.n_tv_splits):
                if use_early_stopping and not has_stopped[tv_split_idx]:
                    if current_epoch > early_stopping_multiplicative_patience \
                            * self.best_val_epochs[tt_split_idx][tv_split_idx] \
                            + early_stopping_additive_patience:
                        has_stopped[tv_split_idx] = True

                if not has_stopped[tv_split_idx]:
                    # compute best validation errors
                    current_err = val_errors[tt_split_idx, tv_split_idx].item()
                    best_err = self.best_val_errors[
                        tt_split_idx][tv_split_idx]
                    # use <= on purpose such that latest epoch among tied best epochs is kept
                    # this has been slightly beneficial for accuracy in previous experiments
                    improved = current_err <= best_err if use_last_best_epoch \
                        else current_err < best_err
                    if improved:
                        self.best_val_errors[tt_split_idx][tv_split_idx] = current_err
                        self.best_val_epochs[tt_split_idx][tv_split_idx] = (
                                self.progress.epoch + 1
                        )

            if not any(has_stopped):
                # compute best mean validation errors (averaged over sub-splits (cv/refit))
                # use <= on purpose such that latest epoch among tied best epochs is kept
                # this has been slightly beneficial for accuracy in previous experiments
                improved = mean_val_errors[tt_split_idx] <= self.best_mean_val_errors[
                    tt_split_idx] if use_last_best_epoch \
                    else mean_val_errors[tt_split_idx] < self.best_mean_val_errors[tt_split_idx]
                if improved:
                    self.best_mean_val_errors[tt_split_idx] = mean_val_errors[tt_split_idx]
                    self.best_mean_val_epochs[tt_split_idx] = (
                            self.progress.epoch + 1
                    )
        self.progress.epoch += 1

        if use_early_stopping and all(sum(self.has_stopped_list, [])):
            self.trainer.should_stop = True

    def on_fit_end(self):
        if self.creator.config.get("use_best_epoch", True):
            self.fit_params = [{'stop_epoch': mean_ep, 'best_indiv_stop_epochs': single_eps}
                               for mean_ep, single_eps in zip(self.best_mean_val_epochs, self.best_val_epochs)]
        else:
            self.fit_params = [
                {"stop_epoch": self.progress.max_epochs}
                for i in range(self.creator.n_tt_splits)
            ]

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        self.model.eval()
        with torch.no_grad():
            return self.model(batch)["x_cont"].to("cpu")

    def configure_optimizers(self):
        param_groups = [{"params": [p], "lr": 0.01} for p in self.model.parameters()]
        return get_opt_class(self.config.get('opt', 'adam'))(param_groups, self.hp_manager)
