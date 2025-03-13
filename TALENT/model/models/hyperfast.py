import os
import math
import torch
import requests
import numpy as np
import pandas as pd
import torch.nn.functional as F
from types import SimpleNamespace
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from TALENT.model.lib.hyperfast.utils import (
    seed_everything,
    transform_data_for_main_network,
    forward_main_network,
    nn_bias_logits,
    fine_tune_main_network,
)
from TALENT.model.lib.hyperfast.model import HyperFast

# Source: https://github.com/AI-sandbox/HyperFast/blob/main/hyperfast/hyperfast.py

class HyperFastClassifier(BaseEstimator):
    """
    A scikit-learn-like interface for the HyperFast model.

    Attributes:
        device (str): Device to run the model on.
        n_ensemble (int): Number of ensemble models to use.
        batch_size (int): Size of the batch for weight prediction and ensembling.
        nn_bias (bool): Whether to use nearest neighbor bias.
        optimization (str): Strategy for optimization, can be None, 'optimize', or 'ensemble_optimize'.
        optimize_steps (int): Number of optimization steps.
        torch_pca (bool): Whether to use PyTorch-based PCA optimized for GPU (fast) or scikit-learn PCA (slower).
        seed (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        device,
        seed,
        n_ensemble=1, # 1 or 16
        batch_size=2048,
        nn_bias=False,
        optimization="None", # "None" or "ensemble_optimize"
        optimize_steps=64,
        torch_pca=True,
        config={
            "hn_n_layers": 4,
            "hn_hidden_size": 1024,
            "clip_data_value": 27.6041,
            "rf_size": 2**15,
            "n_dims": 784,
            "main_n_layers": 3,
            "max_categories": 46,
            "lr": 0.0001,
            "model_url": "https://figshare.com/ndownloader/files/43484094",
            "model_path": "model/models/hyperfast/hyperfast.ckpt",
        }
    ):
        self.device = device
        self.n_ensemble = n_ensemble
        self.batch_size = batch_size
        self.nn_bias = nn_bias
        self.optimization = optimization
        self.optimize_steps = optimize_steps
        self.torch_pca = torch_pca
        self.seed = seed

        seed_everything(self.seed)
        self._cfg = self._load_config(config, self.device, self.torch_pca, self.nn_bias)
        self._model = self._initialize_model(self._cfg)

    def _load_config(self, config, device, torch_pca, nn_bias):
        cfg = SimpleNamespace(**config)
        cfg.device = device
        cfg.torch_pca = torch_pca
        cfg.nn_bias = nn_bias
        return cfg

    def _initialize_model(self, cfg):
        model = HyperFast(cfg).to(cfg.device)
        if not os.path.exists(cfg.model_path):
            self._download_model(cfg.model_url, cfg.model_path)

        try:
            # print(f"Loading model from {cfg.model_path}...", flush=True)
            model.load_state_dict(
                torch.load(cfg.model_path, map_location=torch.device(cfg.device))
            )
            # print(f"Model loaded from {cfg.model_path}", flush=True)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found at {cfg.model_path}") from e
        model.eval()
        return model

    def _download_model(self, url, local_path):
        print(
            f"Downloading model from {url}, since no model was found at {local_path}",
            flush=True,
        )
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
            print(f"Model downloaded and saved to {local_path}")
        else:
            raise ConnectionError(f"Failed to download the model from {url}")

    def _preprocess_fitting_data(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype=np.float32).copy()
        y = np.array(y, dtype=np.int64).copy()
        # Impute missing values for numerical features with the mean
        self._num_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        self._all_feature_idxs = np.arange(x.shape[1])
        self._numerical_feature_idxs = np.setdiff1d(
            self._all_feature_idxs, self._cat_features
        )
        if len(self._numerical_feature_idxs) > 0:
            self._num_imputer.fit(x[:, self._numerical_feature_idxs])
            x[:, self._numerical_feature_idxs] = self._num_imputer.transform(
                x[:, self._numerical_feature_idxs]
            )

        if len(self._cat_features) > 0:
            # Impute missing values for categorical features with the most frequent category
            self.cat_imputer = SimpleImputer(
                missing_values=np.nan, strategy="most_frequent"
            )
            self.cat_imputer.fit(x[:, self._cat_features])
            x[:, self._cat_features] = self.cat_imputer.transform(
                x[:, self._cat_features]
            )

            # One-hot encode categorical features
            x = pd.DataFrame(x)
            self.one_hot_encoder = ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                        self._cat_features,
                    )
                ],
                remainder="passthrough",
            )
            self.one_hot_encoder.fit(x)
            x = self.one_hot_encoder.transform(x)

        # Standardize data
        self._scaler = StandardScaler()
        self._scaler.fit(x)
        x = self._scaler.transform(x)
        return torch.tensor(x, dtype=torch.float).to(self.device), torch.tensor(
            y, dtype=torch.long
        ).to(self.device)

    def _preprocess_test_data(self, x_test):
        x_test = np.array(x_test, dtype=np.float32).copy()
        # Impute missing values for numerical features with the mean
        if len(self._numerical_feature_idxs) > 0:
            x_test[:, self._numerical_feature_idxs] = self._num_imputer.transform(
                x_test[:, self._numerical_feature_idxs]
            )

        if len(self._cat_features) > 0:
            # Impute missing values for categorical features with the most frequent category
            x_test[:, self._cat_features] = self.cat_imputer.transform(
                x_test[:, self._cat_features]
            )

            # One-hot encode categorical features
            x_test = pd.DataFrame(x_test)
            x_test = self.one_hot_encoder.transform(x_test)

        # Standardize data
        x_test = self._scaler.transform(x_test)
        return x_test

    def _initialize_fit_attributes(self):
        self._rfs = []
        self._pcas = []
        self._main_networks = []
        self._X_preds = []
        self._y_preds = []

    def _sample_data(self, X, y):
        indices = torch.randperm(len(X))[: self.batch_size]
        X_pred, y_pred = X[indices].flatten(start_dim=1), y[indices]
        if X_pred.shape[0] < self._cfg.n_dims:
            n_repeats = math.ceil(self._cfg.n_dims / X_pred.shape[0])
            X_pred = torch.repeat_interleave(X_pred, n_repeats, axis=0)
            y_pred = torch.repeat_interleave(y_pred, n_repeats, axis=0)
        return X_pred, y_pred

    def _store_network(self, rf, pca, main_network, X_pred, y_pred):
        self._rfs.append(rf)
        self._pcas.append(pca)
        self._main_networks.append(main_network)
        self._X_preds.append(X_pred)
        self._y_preds.append(y_pred)

    def fit(self, X, y, cat_features=[]):
        """
        Generates a main model for the given data.

        Args:
            X (array-like): Input features.
            y (array-like): Target values.
            cat_features (list, optional): List of categorical features. Defaults to an empty list.
        """
        seed_everything(self.seed)
        X, y = check_X_y(X, y)
        self._cat_features = cat_features
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        X, y = self._preprocess_fitting_data(X, y)
        self._initialize_fit_attributes()

        for n in range(self.n_ensemble):
            X_pred, y_pred = self._sample_data(X, y)
            self.n_classes_ = len(torch.unique(y_pred).cpu().numpy())

            rf, pca, main_network = self._model(X_pred, y_pred, self.n_classes_)

            if self.optimization == "ensemble_optimize":
                rf, pca, main_network, self._model.nn_bias = fine_tune_main_network(
                    self._cfg,
                    X_pred,
                    y_pred,
                    self.n_classes_,
                    rf,
                    pca,
                    main_network,
                    self._model.nn_bias,
                    self.device,
                    self.optimize_steps,
                    self.batch_size,
                )
            self._store_network(rf, pca, main_network, X_pred, y_pred)

        if self.optimization == "optimize" and self.optimize_steps > 0:
            assert len(self._main_networks) == 1
            (
                self._rfs[0],
                self._pcas[0],
                self._main_networks[0],
                self._model.nn_bias,
            ) = fine_tune_main_network(
                self._cfg,
                X,
                y,
                self.n_classes_,
                self._rfs[0],
                self._pcas[0],
                self._main_networks[0],
                self._model.nn_bias,
                self.device,
                self.optimize_steps,
                self.batch_size,
            )

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X = self._preprocess_test_data(X)
        with torch.no_grad():
            X = torch.Tensor(X).to(self.device)
            orig_X = X
            yhats = []
            for jj in range(len(self._main_networks)):
                main_network = self._main_networks[jj]
                rf = self._rfs[jj]
                pca = self._pcas[jj]
                X_pred = self._X_preds[jj]
                y_pred = self._y_preds[jj]

                X_transformed = transform_data_for_main_network(
                    X=X, cfg=self._cfg, rf=rf, pca=pca
                )
                outputs, intermediate_activations = forward_main_network(
                    X_transformed, main_network
                )

                if self.nn_bias:
                    X_pred_ = transform_data_for_main_network(
                        X=X_pred, cfg=self._cfg, rf=rf, pca=pca
                    )
                    outputs_pred, intermediate_activations_pred = forward_main_network(
                        X_pred_, main_network
                    )
                    for bb, bias in enumerate(self._model.nn_bias):
                        if bb == 0:
                            outputs = nn_bias_logits(
                                outputs, orig_X, X_pred, y_pred, bias, self.n_classes_
                            )
                        elif bb == 1:
                            outputs = nn_bias_logits(
                                outputs,
                                intermediate_activations,
                                intermediate_activations_pred,
                                y_pred,
                                bias,
                                self.n_classes_,
                            )

                predicted = F.softmax(outputs, dim=1)
                yhats.append(predicted)

            yhats = torch.stack(yhats)
            yhats = torch.sum(yhats, axis=0)
            return yhats.cpu().numpy()

    def predict(self, X):
        outputs = self.predict_proba(X)
        y_pred = np.argmax(outputs, axis=1)
        return y_pred, outputs
