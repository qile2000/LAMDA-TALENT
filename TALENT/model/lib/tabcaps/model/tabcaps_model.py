from dataclasses import dataclass, field
from typing import List, Any, Dict
import torch.cuda
import pandas as pd
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from TALENT.model.lib.tabcaps.lib.utils import FastTensorDataLoader
from qhoptim.pyt import QHAdam
import numpy as np
from abc import abstractmethod
from TALENT.model.lib.tabcaps.lib.utils import (
    PredictDataset,
    validate_eval_set,
    create_dataloaders,
    define_device,
)
from TALENT.model.lib.tabcaps.lib.callbacks import (
    CallbackContainer,
    History,
    EarlyStopping,
    LRSchedulerCallback
)
from TALENT.model.lib.tabcaps.lib.logger import Train_Log
from TALENT.model.lib.tabcaps.lib.metrics import MetricContainer, check_metrics
from TALENT.model.lib.tabcaps.model.capsule_loss import MarginLoss
from TALENT.model.lib.tabcaps.model.tab_capsulenet import CapsuleClassifier, ReconstructCapsNet
from sklearn.base import BaseEstimator
from sklearn.utils import check_array

import warnings


@dataclass
class TabCapsModel(BaseEstimator):
    """ Class for TabCapsModel model.
        Code Architecture modify from Source: https://github.com/dreamquark-ai/tabnet
    """
    decode: bool = False
    mean: int = None
    std: int = None
    sub_class: int = 1
    init_dim: int = None
    primary_capsule_size: int = 16
    digit_capsule_size: int = 16
    leaves: int = 32
    seed: int = 0
    verbose: int = 1
    optimizer_fn: Any = QHAdam
    optimizer_params: Dict = field(default_factory=lambda: dict(lr=2e-2, weight_decay=1e-5, nus=(0.8, 1.0)))
    scheduler_fn: Any = torch.optim.lr_scheduler.StepLR
    scheduler_params: Dict = field(default_factory=lambda: dict(gamma=0.95, step_size=20))
    input_dim: int = None
    output_dim: int = None
    device_name: str = "auto"

    def __post_init__(self):
        self.batch_size = 1024
        self.virtual_batch_size = 256
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # Defining device
        self.device = torch.device(define_device(self.device_name))
        if self.verbose != 0:
            print(f"Device used : {self.device}")

    def __update__(self, **kwargs):
        """
        Updates parameters.
        If does not already exists, creates it.
        Otherwise overwrite with warnings.
        """
        update_list = [
            "input_dim",
            "capsule_num",
            "base_outdim",
            "n_path",
            "mean",
            "std",
        ]
        for var_name, value in kwargs.items():
            if var_name in update_list:
                try:
                    exec(f"global previous_val; previous_val = self.{var_name}")
                    if previous_val != value:  # noqa
                        wrn_msg = f"Pretraining: {var_name} changed from {previous_val} to {value}"  # noqa
                        warnings.warn(wrn_msg)
                        exec(f"self.{var_name} = value")
                except AttributeError:
                    exec(f"self.{var_name} = value")

    def fit(
        self,
        X_train,
        y_train,
        eval_set=None,
        eval_name=None,
        eval_metric=None,
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=256,
        callbacks=None,
        logname=None,
        resume_dir=None,
        device_id=None,
        cfg=None
    ):
        """Train a neural network stored in self.network
        Using train_dataloader for training data and
        valid_dataloader for validation.

        Parameters
        ----------
        X_train : np.ndarray
            Train set
        y_train : np.array
            Train targets
        eval_set : list of tuple
            List of eval tuple set (X, y).
            The last one is used for early stopping
        eval_name : list of str
            List of eval set names.
        eval_metric : list of str
            List of evaluation metrics.
            The last metric is used for early stopping.
        loss_fn : callable or None
            a PyTorch loss function
        max_epochs : int
            Maximum number of epochs during training
        patience : int
            Number of consecutive non improving epoch before early stopping
        batch_size : int
            Training batch size
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization (virtual_batch_size < batch_size)
        callbacks : list of callback function
            List of custom callbacks
        logname: str
            Setting log name
        resume_dir: str
            The resume file directory
        gpu_id: str
            Single GPU or Multi GPU ID
        """
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.input_dim = X_train.shape[1]
        self._stop_training = False
        self.log = Train_Log(logname, resume_dir, cfg) if (logname or resume_dir) else None
        self.device_id = device_id
        eval_set = eval_set if eval_set else []

        check_array(X_train)

        self.update_fit_params(X_train, y_train, eval_set)
        # Validate and reformat eval set depending on training data
        eval_names, eval_set = validate_eval_set(eval_set, eval_name, X_train, y_train)

        train_dataloader, valid_dataloaders = self._construct_loaders(X_train, y_train, eval_set)

        self._set_network()
        self._set_metrics(eval_metric, eval_names)
        self._set_optimizer()
        self._set_callbacks(callbacks)
        start_epoch = 1
        # best_value = -float('inf') if self._task == 'classification' else float('inf')
        if resume_dir:
            start_epoch, self.network, self._optimizer, best_value, best_epoch = self.log.load_checkpoint(self._optimizer)
        # Call method on_train_begin for all callbacks
        self._callback_container.on_train_begin()
        print("===> Start training ...")
        for epoch_idx in range(start_epoch, self.max_epochs + 1):
            self.epoch = epoch_idx
            # Call method on_epoch_begin for all callbacks
            self._callback_container.on_epoch_begin(epoch_idx)
            
            self._train_epoch(train_dataloader)

            # Apply predict epoch to all eval sets
            for eval_name, valid_dataloader in zip(eval_names, valid_dataloaders):
                # print("eval_name:", eval_name)
                self._predict_epoch(eval_name, valid_dataloader)

            # Call method on_epoch_end for all callbacks
            self._callback_container.on_epoch_end(epoch_idx, logs=self.history.epoch_metrics)

            # self.save_check()
            print('LR: ' + str(self._optimizer.param_groups[0]['lr']))
            if self._stop_training:
                break

        # Call method on_train_end for all callbacks
        self._callback_container.on_train_end()
        loss = self.history.epoch_metrics['loss']
        return None, loss, None

    def predict(self, X, y, decode=False):
        """
        Make predictions on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor`
            Input data

        Returns
        -------
        predictions : np.array
        """
        self.network.eval()
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        dataloader = FastTensorDataLoader(X, y, batch_size=self.batch_size, shuffle=False)
        reconstruct_data = []
        y_list = []
        pred = []
        print('===> Starting test ... ')
        for batch_nb, (data, label) in enumerate(dataloader):
            data = data.to(self.device).float()
            label = label.to(self.device).long()
            with torch.no_grad():
                if decode == True:
                    y_one_hot = F.one_hot(label, self.output_dim).float()
                    output, reconstruction = self.network(data, y_one_hot)
                    reconstruct_data.append(reconstruction.cpu().detach().numpy())
                else:
                    output = self.network(data)

                y_list.append(label.cpu().detach().numpy())
                pred.append(output.cpu().detach().numpy())

        if decode == True:
            reconstruct_data = np.vstack(reconstruct_data)
            y_list = np.hstack(y_list)
            reconstruct_data = np.concatenate([reconstruct_data, y_list[:, None]], axis=1)
            reconstruct_data = pd.DataFrame(reconstruct_data)
        y_true, y_pred = self.stack_batches(y_list, pred)
        return y_true, y_pred, reconstruct_data

    def save_check(self, path, seed):
        save_dict = {
            'epoch': self.epoch,
            'model': self.network,
            # 'state_dict': self.network.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'best_value': self._callback_container.callbacks[1].best_loss,
            "best_epoch": self._callback_container.callbacks[1].best_epoch
        }
        torch.save(save_dict, path + f'/epoch-last-{seed}.pth')

    def load_model(self, filepath, input_dim, output_dim):
        """Load model.

        Parameters
        ----------
        filepath : str
            Path of the model.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        load_model = torch.load(filepath)
        self.network = load_model['model']

        self.network.eval()
        return

    def _train_epoch(self, train_loader):
        """
        Trains one epoch of the network in self.network

        Parameters
        ----------
        train_loader : a :class: `torch.utils.data.Dataloader`
            DataLoader with train set
        """
        self.network.train()
        loss, recon_metric, recon_mse = [], [], []
        for batch_idx, (batch) in enumerate(train_loader):
            X, y = batch[0], batch[1]
            y_distribution = batch[2] if len(batch) == 3 else None
            # print(y_distribution)
            self._callback_container.on_batch_begin(batch_idx)

            batch_logs = self._train_batch(X, y)

            self._callback_container.on_batch_end(batch_idx, batch_logs)
            loss.append(batch_logs['loss'])
        epoch_logs = {"lr": self._optimizer.param_groups[-1]["lr"], "loss": np.mean(loss)}

        self.history.epoch_metrics.update(epoch_logs)
        return

    def _train_batch(self, X, y):
        """
        Trains one batch of data

        Parameters
        ----------
        X : torch.Tensor
            Train matrix
        y : torch.Tensor
            Target matrix

        Returns
        -------
        batch_outs : dict
            Dictionnary with "y": target and "score": prediction scores.
        batch_logs : dict
            Dictionnary with "batch_size" and "loss".
        """
        batch_logs = {"batch_size": X.shape[0]}
        X = X.to(self.device).float()
        y = y.to(self.device).long()
        self._optimizer.zero_grad()
        y_one_hot = F.one_hot(y, self.output_dim).float()

        recon_loss = 0.
        if self.decode:
            pred, reconstruction = self.network(X, y_one_hot)
            recon_loss = F.mse_loss(reconstruction, X, reduction='sum')
        else:
            pred = self.network(X)

        main_loss = self.loss_fn(pred, y_one_hot)
        loss = main_loss + recon_loss * 0.000001
        # Perform backward pass and optimization
        loss.backward()
        self._optimizer.step()

        batch_logs["loss"] = loss.cpu().detach().numpy().item()
        return batch_logs

    def _predict_epoch(self, name, loader):
        """
        Predict an epoch and update metrics.

        Parameters
        ----------
        name : str
            Name of the validation set
        loader : torch.utils.data.Dataloader
                DataLoader with validation set
        """
        # Setting network on evaluation mode
        self.network.eval()
        list_y_true = []
        list_y_score = []
        # recon_metric, recon_mse = [], []
        # Main loop
        for batch_idx, batch in enumerate(loader):
            X, y = batch[0], batch[1]
            scores, batch_logs = self._predict_batch(X, y)
            list_y_true.append(y)
            list_y_score.append(scores)

        y_true, scores = self.stack_batches(list_y_true, list_y_score)

        metrics_logs = self._metric_container_dict[name](y_true, scores)
        # print("scores:", scores)

        self.network.train()
        self.history.epoch_metrics.update(metrics_logs)
        # print("epoch_metrics", self.history.epoch_metrics)
        return

    def _predict_batch(self, X, y):
        """
        Predict one batch of data.

        Parameters
        ----------
        X : torch.Tensor
            Owned products

        Returns
        -------
        np.array
            model scores
        """
        self.network.eval()
        X = X.to(self.device).float()
        y = y.to(self.device).long()
        batch_logs = {}
        y_one_hot = F.one_hot(y, self.output_dim).float()
        with torch.no_grad():
            if self.decode:
                scores, reconstruction = self.network(X, y_one_hot)
            else:
                scores = self.network(X)

            if isinstance(scores, list):
                scores = sum([x.cpu().detach().numpy() for x in scores])
            else:
                scores = scores.cpu().detach().numpy()

        return scores, batch_logs


    @abstractmethod
    def update_fit_params(self, X_train, y_train, eval_set):
        """
        Set attributes relative to fit function.

        Parameters
        ----------
        X_train : np.ndarray
            Train set
        y_train : np.array
            Train targets
        eval_set : list of tuple
            List of eval tuple set (X, y).
        """
        raise NotImplementedError(
            "users must define update_fit_params to use this base class"
        )

    def _set_network(self):
        """Setup the network and explain matrix."""
        print("===> Building model ...")
        self.params = {'out_capsule_num': self.output_dim * self.sub_class,
                  'init_dim': self.init_dim,
                  'primary_capsule_dim': self.primary_capsule_size,
                  'digit_capsule_dim': self.digit_capsule_size,
                  'n_leaves': self.leaves
                  }

        self.loss_fn = MarginLoss()
        self.network = ReconstructCapsNet(self.input_dim, self.output_dim, **self.params) \
            if self.decode else CapsuleClassifier(self.input_dim, self.output_dim, **self.params)
        self.recon_metric = accuracy_score

        if len(self.device_id) > 1:
            self.network = DataParallel(self.network)
        self.network = self.network.to(self.device)

    def _set_metrics(self, metrics, eval_names):
        """Set attributes relative to the metrics.

        Parameters
        ----------
        metrics : list of str
            List of eval metric names.
        eval_names : list of str
            List of eval set names.

        """
        metrics = metrics or [self._default_metric]

        metrics = check_metrics(metrics)
        # Set metric container for each sets
        self._metric_container_dict = {}
        for name in eval_names:
            self._metric_container_dict.update(
                {name: MetricContainer(metrics, prefix=f"{name}_")}
            )

        self._metrics = []
        self._metrics_names = []
        for _, metric_container in self._metric_container_dict.items():
            self._metrics.extend(metric_container.metrics)
            self._metrics_names.extend(metric_container.names)

        # Early stopping metric is the last eval metric

        self.early_stopping_metric = (self._metrics_names[-1] if len(self._metrics_names) > 0 else None)

    def _set_callbacks(self, custom_callbacks):
        """Setup the callbacks functions.

        Parameters
        ----------
        custom_callbacks : list of func
            List of callback functions.

        """
        # Setup default callbacks history, early stopping and scheduler
        callbacks = []
        self.history = History(self, verbose=self.verbose)
        callbacks.append(self.history)
        if (self.early_stopping_metric is not None) and (self.patience > 0):
            early_stopping = EarlyStopping(
                early_stopping_metric=self.early_stopping_metric,
                is_maximize=(
                    self._metrics[-1]._maximize if len(self._metrics) > 0 else None
                ),
                patience=self.patience,
            )
            callbacks.append(early_stopping)
        else:
            print("No early stopping will be performed, last training weights will be used.")

        if self.scheduler_fn is not None:
            # Add LR Scheduler call_back
            is_batch_level = self.scheduler_params.pop("is_batch_level", False)
            scheduler = LRSchedulerCallback(
                scheduler_fn=self.scheduler_fn,
                scheduler_params=self.scheduler_params,
                optimizer=self._optimizer,
                early_stopping_metric=self.early_stopping_metric,
                is_batch_level=is_batch_level,
            )
            callbacks.append(scheduler)

        if custom_callbacks:
            callbacks.extend(custom_callbacks)
        self._callback_container = CallbackContainer(callbacks)
        self._callback_container.set_trainer(self)

    def _set_optimizer(self):
        """Setup optimizer."""
        self._optimizer = self.optimizer_fn(self.network.parameters(), **self.optimizer_params)

    def _construct_loaders(self, X_train, y_train, eval_set):
        """Generate dataloaders for train and eval set.

        Parameters
        ----------
        X_train : np.array
            Train set.
        y_train : np.array
            Train targets.
        eval_set : list of tuple
            List of eval tuple set (X, y).

        Returns
        -------
        train_dataloader : `torch.utils.data.Dataloader`
            Training dataloader.
        valid_dataloaders : list of `torch.utils.data.Dataloader`
            List of validation dataloaders.

        """
        # all weights are not allowed for this type of model
        y_train_mapped = self.prepare_target(y_train)
        for i, batch in enumerate(eval_set):
            if len(batch) == 3:
                eval_set[i] = (batch[0], self.prepare_target(batch[1]), batch[2])
            else:
                eval_set[i] = (batch[0], self.prepare_target(batch[1]))

        train_dataloader, valid_dataloaders = create_dataloaders(
            X_train,
            y_train_mapped,
            eval_set,
            self.batch_size
        )
        return train_dataloader, valid_dataloaders


    def _update_network_params(self):
        self.network.virtual_batch_size = self.virtual_batch_size

    @abstractmethod
    def compute_loss(self, y_score, y_true):
        """
        Compute the loss.

        Parameters
        ----------
        y_score : a :tensor: `torch.Tensor`
            Score matrix
        y_true : a :tensor: `torch.Tensor`
            Target matrix

        Returns
        -------
        float
            Loss value
        """
        raise NotImplementedError(
            "users must define compute_loss to use this base class"
        )

    @abstractmethod
    def prepare_target(self, y):
        """
        Prepare target before training.

        Parameters
        ----------
        y : a :tensor: `torch.Tensor`
            Target matrix.

        Returns
        -------
        `torch.Tensor`
            Converted target matrix.
        """
        raise NotImplementedError(
            "users must define prepare_target to use this base class"
        )