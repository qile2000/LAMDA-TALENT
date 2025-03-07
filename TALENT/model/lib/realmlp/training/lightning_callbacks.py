from typing import List, Any, Optional

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch import Tensor

from TALENT.model.lib.realmlp.nn_models.base import Variable, Layer
from TALENT.model.lib.realmlp.training.coord import HyperparamManager
from TALENT.model.lib.realmlp.training.logging import Logger


class ParamCheckpointer:
    def __init__(self, n_tv_splits, n_tt_splits):
        self.n_tv_splits = n_tv_splits
        self.n_tt_splits = n_tt_splits
        self.ckpt_params = [None] * (self.n_tt_splits * self.n_tv_splits)
        self.ckpt_buffers = [None] * (self.n_tt_splits * self.n_tv_splits)

    def save(self, parallel_idx: int, model_idx: int, model: Layer):
        idx = self.n_tv_splits * parallel_idx + model_idx
        with torch.no_grad():
            for ckpt, values in [(self.ckpt_params, model.parameters()), (self.ckpt_buffers, model.buffers())]:
                if ckpt[idx] is None:
                    ckpt[idx] = [v[idx].clone() for v in values]
                else:
                    for c, v in zip(ckpt[idx], values):
                        c.copy_(v[idx])

    def restore(self, parallel_idx: int, model_idx: int, model: Layer):
        idx = self.n_tv_splits * parallel_idx + model_idx
        with torch.no_grad():
            for ckpt, values in [(self.ckpt_params, model.parameters()), (self.ckpt_buffers, model.buffers())]:
                if ckpt[idx] is not None:
                    for c, v in zip(ckpt[idx], values):
                        # print(f'Restore diff: {v[start:end]-c}')
                        v[idx] = c

    def save_all(self, model: Layer):
        for parallel_idx in range(self.n_tt_splits):
            for model_idx in range(self.n_tv_splits):
                self.save(parallel_idx, model_idx, model)

    def restore_all(self, model: Layer):
        for parallel_idx in range(self.n_tt_splits):
            for model_idx in range(self.n_tv_splits):
                self.restore(parallel_idx, model_idx, model)


class HyperparamCallback(Callback):
    def __init__(self, hp_manager):
        self.hp_manager = hp_manager

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        # print(list(pl_module.model.parameters())[-1][0, -1].item())
        self.hp_manager.update_hypers(pl_module)

    def on_before_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", loss: Tensor) -> None:
        reg_terms = self.hp_manager.reg_terms
        if len(reg_terms) > 0:
            pl_module.loss += sum(reg_terms)


class L1L2RegCallback(Callback):
    def __init__(self, hp_manager: HyperparamManager, model: Layer):
        self.hp_manager = hp_manager
        self.params: List[Variable] = list(model.parameters())
        self.l1_getters = [self.hp_manager.register_hyper('l1_reg', p.context.scope, default=0.0)
                           for p in self.params]
        self.l2_getters = [self.hp_manager.register_hyper('l2_reg', p.context.scope, default=0.0)
                           for p in self.params]

    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for l1_getter, l2_getter, p in zip(self.l1_getters, self.l2_getters, self.params):
            l1_reg = l1_getter() * p.hyper_factors.get('l1_reg', 1.0)
            l2_reg = l2_getter() * p.hyper_factors.get('l2_reg', 1.0)

            if l1_reg != 0.0:
                p.grad += l1_reg * torch.sign(p)
            if l2_reg != 0.0:
                p.grad += (2.0 * l2_reg) * p

        self.hp_manager.update_hypers(pl_module)


class ModelCheckpointCallback(Callback):
    def __init__(self, n_tt_splits: int, n_tv_splits: int, use_best_mean_epoch: bool, restore_best: bool = False):
        self.n_tt_splits = n_tt_splits
        self.n_tv_splits = n_tv_splits
        self.restore_best = restore_best
        self.use_best_mean_epoch = use_best_mean_epoch
        self.ckpt = ParamCheckpointer(n_tv_splits=n_tv_splits, n_tt_splits=self.n_tt_splits)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.ckpt.save_all(pl_module.model)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for tt_split_idx in range(self.n_tt_splits):
            for tv_split_idx in range(self.n_tv_splits):
                if self.use_best_mean_epoch:
                    if pl_module.best_mean_val_epochs[tt_split_idx] == pl_module.progress.epoch:
                        # if this is the best epoch, save the model
                        self.ckpt.save(tt_split_idx, tv_split_idx, pl_module.model)
                else:
                    if pl_module.best_val_epochs[tt_split_idx][tv_split_idx] == pl_module.progress.epoch:
                        # print(f'found improvement')
                        # if this is the best epoch, save the model
                        self.ckpt.save(tt_split_idx, tv_split_idx, pl_module.model)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # print(f'Before restore: {list(pl_module.model.parameters())[-1]}')
        # restore best params
        if not self.restore_best:
            raise RuntimeError('ValidationCallback: Cannot restore best params when using save_best_params=False')
        self.ckpt.restore_all(pl_module.model)
        # print(f'After restore: {list(pl_module.model.parameters())[-1]}')


class StopAtEpochsCallback(Callback):
    def __init__(self, stop_epochs: List[List[int]], n_models: int, model: Layer, logger: Optional[Logger] = None):
        print(f'Refit: {stop_epochs=}')
        self.stop_epochs = stop_epochs
        self.final_stop_epoch = np.max(sum(stop_epochs, []))
        self.model = model
        self.ckpt = ParamCheckpointer(n_tv_splits=n_models, n_tt_splits=len(stop_epochs))
        self.logger = logger
        self.n_models = n_models

    def _handle_epoch(self, trainer: "pl.Trainer", epoch: int) -> None:
        if self.logger:
            self.logger.log(2, f'Refit Epoch {epoch}/{self.final_stop_epoch}')

        if epoch == self.final_stop_epoch:
            # print(f'Stopping the training at epoch {epoch}')
            self.ckpt.restore_all(self.model)
            trainer.should_stop = True
            return

        for tt_split_idx, tv_stop_epochs in enumerate(self.stop_epochs):
            for tv_split_idx, ep in enumerate(tv_stop_epochs):
                if ep == epoch:
                    # print(f'Saving checkpoint for model {i}')
                    self.ckpt.save(tt_split_idx, tv_split_idx, self.model)

    # def on_train_batch_start(
    #     self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    # ) -> None:
    #     print('train batch')

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._handle_epoch(trainer, epoch=0)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._handle_epoch(trainer, epoch=trainer.current_epoch+1)
