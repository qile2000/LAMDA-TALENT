from __future__ import annotations

"""This module contains the input scaling."""

import abc
import dataclasses
import random as random_mod
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union
# Source: https://github.com/younader/dnnr
import numpy as np
import scipy.optimize
import scipy.spatial.distance
import sklearn.base

# from sklearn.metrics import mean_absolute_error, mean_squared_error
import sklearn.metrics as sk_metrics
import tqdm.auto as tqdm
from sklearn import model_selection

import dataclasses
from typing import Any, Callable, Optional, Union

import numpy as np
import sklearn.base

from . import nn_index
from .solver import Solver, create_solver

@dataclasses.dataclass
class NeighborPrediction:
    neighbor_x: np.ndarray
    neighbor_y: np.ndarray
    neighbors_xs: np.ndarray
    neighbors_ys: np.ndarray
    query: np.ndarray  # point to predict
    local_prediction: np.ndarray  # local prediction
    derivative: np.ndarray  # derivative used to predict the point
    prediction_fn: Callable[[np.ndarray], np.ndarray]
    intercept: Optional[np.ndarray] = None


@dataclasses.dataclass
class DNNRPrediction:
    query: np.ndarray
    y_pred: np.ndarray
    neighbor_predictions: list[NeighborPrediction]
    y_true: Optional[np.ndarray] = None


@dataclasses.dataclass
class DNNR(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """DNNR model class.

    metric: distance metric used in the nearest neighbor index

    # Order of the Approximation

    The order of approximation can be controlled with the `order` argument:

        - `1`: Uses first-order approximation (the gradient)
        - `2diag`: First-order and diagonal of the second-order derivatives
        - `2`: The first-order and full second-order matrix (gradient & Hessian)
        - `3diag`: First-order and diagonals of the second and third-orders

    The recommendation is to use `1` which is the most efficient one, using
    `2diag` can sometimes improve the performance. `2` and `3diag` are more
    expensive and usually also do not deliver a better performance.

    Args:
        n_neighbors: number of nearest neighbors to use. The default value of
            `3` is usually a good choice.
        n_derivative_neighbors: number of neighbors used in approximating the
            derivatives. As a default value, we choose `3 * dim` where `dim` is
            the dimension of the input data. This is usually a good heuristic,
            but we would recommend to use a hyperparameter search to find the
            best value for it.
        order: Taylor approximation order, one of `1`, `2`, `2diag`, `3diag`.
            The preferable option here is `1` and sometimes `2diag` can deliver
            small improvements. `2` and `3diag` are implemented but usually do
            not yield significant improvements.
        fit_intercept: if True, the intercept is estimated. Otherwise, the
            point's ground truth label is used.
        solver: name of the equation solver used to approximate the derivatives.
            As default `linear_regression` is used. Other options are
            `scipy_lsqr`, `numpy`, `ridge` and `lasso`. Also accepts any class
            inheriting from `dnnr.solver.Solver`.
        index: name of the index to be used for nearest neighbor (`annoy` or
            `kd_tree`). Also accepts any subclass of `dnnr.nn_index.BaseIndex`.
        index_kwargs: keyword arguments passed to the index constructor.
        scaling: name of the scaling method to be used. If it is `None` or
            `no_scaling`, the data is not scaled. If it is `learned`, the
            scaling is learned using the cosine similarity objective.
        scaling_kwargs: keyword arguments to be passed to the scaling method.
        precompute_derivatives: if True, the gradient is computed for each
            training point during the `fit`. Otherwise, the gradient is computed
            during the prediction.
        clip: whether to clip the predicted output to the maximum and
            minimum of the target values of the train set: `[y_min, y_max]`.
    """

    n_neighbors: int = 3
    n_derivative_neighbors: int = -1
    order: str = "1"
    fit_intercept: bool = False
    solver: Union[str, Solver] = "linear_regression"
    index: Union[str, nn_index.BaseIndex] = "annoy"
    index_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    scaling: Union[None, str, InputScaling] = "learned"
    scaling_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    precompute_derivatives: bool = False
    clip: bool = False

    def __post_init__(self):

        self.nn_index: nn_index.BaseIndex

        if isinstance(self.index, str):
            index_cls = nn_index.get_index_class(self.index)
            self.nn_index = index_cls(**self.index_kwargs)
        else:
            self.nn_index = self.index

        self.derivatives_: Optional[list[np.ndarray]] = None

        self._check_valid_order(self.order)

        self.fitted_ = False

    def _precompute_derivatives(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> None:
        """Computes the gradient for the training points and their estimated
        label from the taylor expansion

        Args:
            X_train (np.ndarray) with shape (n_samples, n_features)
            y_train (np.ndarray) with shape (n_samples, 1)
        """
        self.derivatives_ = []

        for v in X_train:
            indices, _ = self.nn_index.query_knn(
                v, self.n_derivative_neighbors + 1
            )
            # ignore the first index as its the queried point itself
            indices = indices[1:]

            self.derivatives_.append(
                self._estimate_derivatives(X_train[indices], y_train[indices])
            )

    def _get_scaler(self) -> InputScaling:
        """Returns the scaler object"""
        if self.scaling is None:
            return NoScaling()
        elif isinstance(self.scaling, str):
            if self.scaling in ["None", 'no_scaling']:
                return NoScaling()
            elif self.scaling == "learned":
                return LearnedScaling(**self.scaling_kwargs)
            else:
                raise ValueError("Unknown scaling method")
        else:
            return self.scaling

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> DNNR:

        # save dataset shapes
        m, n = X_train.shape
        self.n = n
        self.m = m

        if self.n_derivative_neighbors == -1:
            self.n_derivative_neighbors = 3 * self.n

        if isinstance(self.solver, str):
            self.solver_ = create_solver(self.solver)
        else:
            self.solver_ = self.solver
        # create and build the nearest neighbors index

        # save a copy of the training data, should be only used
        # with precompute_derivatives=False
        self.max_y = np.max(y_train)
        self.min_y = np.min(y_train)

        self.scaler_ = self._get_scaler()
        # scale the training data
        self.X_train = self.scaler_.fit_transform(X_train, y_train)
        del X_train
        self.y_train = y_train

        self.nn_index.fit(self.X_train, **self.index_kwargs)

        if self.precompute_derivatives:
            self._precompute_derivatives(self.X_train, y_train)

        self.fitted_ = True
        return self

    def _check_valid_order(self, order: str) -> None:
        if order not in ["1", "2", "2diag", "3diag"]:
            raise ValueError(
                "Unknown order. Must be one of `1`, `2`, `2diag`, `3diag`"
            )

    def _compute_deltas(
        self, query: np.ndarray, xs: np.ndarray, order: str
    ) -> np.ndarray:
        self._check_valid_order(order)

        def _create_2der_mat(mat: np.ndarray) -> np.ndarray:
            """Creates 2-order matrix."""

            der_mat = np.zeros((mat.shape[0], mat.shape[1] ** 2))
            for i in range(mat.shape[0]):
                der_mat[0, :] = (
                    mat[i].reshape(-1, 1) @ mat[i].reshape(-1, 1).T
                ).reshape(-1)
            return der_mat

        deltas_1st = xs - query

        if self.fit_intercept:
            deltas_1st = np.concatenate(
                [deltas_1st, np.ones((deltas_1st.shape[0], 1))], axis=1
            )

        if "1" == order:
            deltas = deltas_1st
        # take care of higher order terms
        elif "2diag" == order:
            deltas_2nd = 0.5 * np.power(xs - query, 2)
            deltas = np.concatenate([deltas_1st, deltas_2nd], axis=1)
        elif "2" == order:
            deltas_2nd = _create_2der_mat(xs - query)
            deltas = np.concatenate([deltas_1st, deltas_2nd], axis=1)
        elif "3diag" == order:
            deltas_2nd = 0.5 * np.power(xs - query, 2)
            deltas_3rd = (1 / 6) * np.power(xs - query, 3)
            deltas = np.concatenate(
                [deltas_1st, deltas_2nd, deltas_3rd], axis=1
            )
        else:
            raise ValueError(f"Unknown order: {order}")
        return deltas

    def _estimate_derivatives(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_neighbors: Optional[int] = None,
        order: Optional[str] = None,
    ) -> np.ndarray:

        nn_indices, _ = self.nn_index.query_knn(
            x, n_neighbors or self.n_derivative_neighbors
        )
        ys = self.y_train[nn_indices] - y
        order = order or self.order

        deltas = self._compute_deltas(x, self.X_train[nn_indices], order)
        w = np.ones(deltas.shape[0])
        # solve for the gradients nn_y_hat
        gamma = self.solver_.solve(deltas, ys, w)
        return gamma

    def _compute_local_prediction(
        self,
        query: np.ndarray,
        neighbor: np.ndarray,
        derivatives: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        intercept = derivatives[self.n] if self.fit_intercept else y
        x_delta = query - neighbor

        # perform taylor approximation to predict the point's label
        prediction = intercept + derivatives[: self.n].dot(x_delta)
        offset = 1 if self.fit_intercept else 0
        # take care of higher order terms:
        if "2diag" in self.order:
            nn_y_hat_2nd = derivatives[self.n + offset :]
            prediction += nn_y_hat_2nd.dot(0.5 * (np.power(x_delta, 2)))
        elif "2" in self.order:
            nn_y_hat_2nd = derivatives[self.n + offset :]
            nn_y_hat_2nd = nn_y_hat_2nd.reshape(self.n, self.n)
            prediction += 0.5 * (x_delta).T.dot(nn_y_hat_2nd).dot(x_delta)
        elif "3diag" in self.order:
            nn_y_hat_2nd = derivatives[self.n + offset : 2 * self.n + offset]
            nn_y_hat_3rd = derivatives[2 * self.n + offset :]
            prediction = (
                prediction
                + nn_y_hat_2nd.dot(0.5 * (np.power(x_delta, 2)))
                + nn_y_hat_3rd.dot((1 / 6) * (np.power(x_delta, 3)))
            )
        return prediction

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("DNNR is not fitted! Call `.fit()` first.")

        predictions = []
        for v in self.scaler_.transform(X_test):
            indices, _ = self.nn_index.query_knn(v, self.n_neighbors)
            predictions_of_neighbors = []
            for i in range(self.n_neighbors):
                # get the neighbor's neighbors' features and labels
                nn = self.X_train[indices[i]]
                nn_y = self.y_train[indices[i]]
                # gamma contains all estimated derivatives
                if self.derivatives_ is not None:
                    gamma = self.derivatives_[int(indices[i])]
                else:
                    gamma = self._estimate_derivatives(nn, nn_y)

                prediction = self._compute_local_prediction(v, nn, gamma, nn_y)
                predictions_of_neighbors.append(prediction)
            predictions.append(np.mean(predictions_of_neighbors))
        if self.clip:
            return np.clip(predictions, a_min=self.min_y, a_max=self.max_y)
        return np.array(predictions)

    def point_analysis(
        self,
        X_test: np.ndarray,
        y_test: Optional[np.ndarray] = None,
    ) -> list[DNNRPrediction]:
        index = 0
        predictions = []
        for v in X_test:
            neighbors = []
            indices, _ = self.nn_index.query_knn(v, self.n_neighbors + 1)

            # if point is in the training set, we skip it
            if np.allclose(v, self.X_train[indices[0]]):
                indices = indices[1:]
            else:
                indices = indices[:-1]

            for i in range(self.n_neighbors - 1):
                # get the neighbhor's features and label
                nn = self.X_train[indices[i]]
                nn_y = self.y_train[indices[i]]
                # get the neighbhors of this neighbhor
                nn_indices, _ = self.nn_index.query_knn(
                    nn, self.n_derivative_neighbors
                )
                nn_indices = nn_indices[1:]  # drop the neighbor itself
                # Δx = X_{nn} - X_{i}
                gamma = self._estimate_derivatives(nn, nn_y)

                local_pred = self._compute_local_prediction(v, nn, gamma, nn_y)
                neighbor_pred = NeighborPrediction(
                    neighbor_x=nn,
                    neighbor_y=nn_y,
                    neighbors_xs=self.X_train[nn_indices],
                    neighbors_ys=self.y_train[nn_indices],
                    query=v,
                    local_prediction=local_pred,
                    derivative=gamma,
                    intercept=nn_y if self.fit_intercept else gamma[self.n],
                    prediction_fn=lambda query: self._compute_local_prediction(
                        query, self.X_train[nn_indices], gamma, nn_y
                    ),
                )
                neighbors.append(neighbor_pred)
            predictions.append(
                DNNRPrediction(
                    query=v,
                    y_pred=np.mean([n.local_prediction for n in neighbors]),
                    y_true=y_test[index] if y_test is not None else None,
                    neighbor_predictions=neighbors,
                )
            )
            index += 1
        return predictions


class InputScaling(sklearn.base.BaseEstimator, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Returns the scaling vector of the input.

        Args:
            X_train: The training data.
            y_train: The training targets.
            X_test: The test data.
            y_test: The test targets.

        Returns:
            The scaling vector.
        """

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        y = y.reshape(-1)
        self.fit(X, y)
        return self.transform(X)

    @abc.abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms the input.

        Args:
            X: The input.

        Returns:
            The transformed input.
        """


class NoScaling(InputScaling):
    """This class does not scale the input."""

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return np.ones(X_train.shape[1])

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


class _Optimizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step(self, gradients: List[np.ndarray]) -> None:
        """Updates the parameters.

        Args:
            gradients: The gradients of the parameters.
        """


@dataclasses.dataclass
class SGD(_Optimizer):
    """Stochastic gradient descent optimizer.

    Args:
        parameters: The parameters to optimize.
        lr: The learning rate.
    """

    parameters: List[np.ndarray]
    lr: float = 0.01

    def step(self, gradients: List[np.ndarray]) -> None:
        for param, grad in zip(self.parameters, gradients):
            param -= self.lr * grad


@dataclasses.dataclass
class RMSPROP:
    """The RMSPROP optimizer.

    Args:
        parameters: The parameters to optimize.
        lr: The learning rate.
        γ: The decay rate.
        eps: The epsilon to avoid division by zero.
    """

    parameters: List[np.ndarray]
    lr: float = 1e-4
    γ: float = 0.99
    eps: float = 1e-08

    def __post_init__(self):
        self.v = [np.zeros_like(param) for param in self.parameters]

    def step(self, gradients: List[np.ndarray]) -> None:
        for param, grad, v in zip(self.parameters, gradients, self.v):
            # inplace update
            v[:] = self.γ * v + (1 - self.γ) * grad**2
            update = self.lr * grad / (np.sqrt(v) + self.eps)
            param -= update


@dataclasses.dataclass
class LearnedScaling(InputScaling):
    """This class handles the scaling of the input.

    Args:
        n_epochs: The number of epochs to train the scaling.
        optimizer: The optimizer to use (either `SGD` or `RMSPROP`).
        optimizer_params: The parameters of the optimizer.
        epsilon: The epsilon for gradient computation.
        random: The `random.Random` instance for this class.
        show_progress: Whether to show a progress bar.
        fail_on_nan: Whether to fail on NaN values.
    """

    n_epochs: int = 1
    optimizer: Union[str, Type[_Optimizer]] = SGD
    optimizer_params: Dict[str, Any] = dataclasses.field(default_factory=dict)
    shuffle: bool = True
    epsilon: float = 1e-6
    random: random_mod.Random = dataclasses.field(
        default_factory=lambda: random_mod.Random(
            random_mod.randint(0, 2**32 - 1)
        )
    )
    show_progress: bool = False
    fail_on_nan: bool = False
    index: Union[str, Type[nn_index.BaseIndex]] = 'annoy'
    index_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.scaling_: Optional[np.ndarray] = None
        self.scaling_history: list = []
        self.scores_history: list = []
        self.costs_history: list = []
        self.index_cls = nn_index.get_index_class(self.index)
        self._fitted: bool = False

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self.scaling_ is None:
            raise RuntimeError("Not fitted")
        return X * self.scaling_

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        val_size: Optional[int] = None,
    ) -> np.ndarray:
        """Fits the scaling vector.

        Args:
            X_train: The training data.
            y_train: The training targets.
            X_val: The validation data.
            y_val: The validation targets.
            val_size: The size of the validation set.

        If the validation set is not provided, the training set is split into
        a validation set using the `val_size` parameter.

        Returns:
            The scaling vector.
        """

        n_features = X_train.shape[1]
        batch_size = 8 * n_features
        scaling = np.ones((1, n_features))

        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be either given or not.")

        if X_val is None and y_val is None:
            split_size = (
                val_size if val_size is not None else int(0.1 * len(X_train))
            )
            if split_size < 10:
                warnings.warn(
                    "Validation split for scaling is small! Scaling is skipped!"
                    f" Got {split_size} samples."
                )
                # do not scale
                self.scaling_ = scaling
                self._fitted = True
                return scaling
            X_train, X_val, y_train, y_val = model_selection.train_test_split(
                X_train,
                y_train,
                test_size=split_size,
                random_state=self.random.randint(0, 2**32 - 1),
            )

        assert X_val is not None
        assert y_val is not None

        def score():
            assert X_val is not None
            n_derivative_neighbors = min(
                int(X_train.shape[0] / 2), X_train.shape[1] * 6
            )
            model = DNNR(
                n_derivative_neighbors=n_derivative_neighbors, scaling=None
            )
            model.fit(scaling * X_train, y_train)
            return sk_metrics.r2_score(y_val, model.predict(scaling * X_val))

        def handle_possible_nans(grad: np.ndarray) -> bool:
            if not np.isfinite(grad).all():
                if self.fail_on_nan:
                    raise RuntimeError("Gradient contains NaN or Inf")

                warnings.warn(
                    "Found inf/nans in gradient. " "Scaling is returned now."
                )

                self.scaling_ = self.scaling_history[
                    np.argmax(self.scores_history)
                ]
                return True
            else:
                return False

        def get_optimizer() -> _Optimizer:
            if isinstance(self.optimizer, str):
                optimizer_cls = {
                    'sgd': SGD,
                    'rmsprop': RMSPROP,
                }[self.optimizer.lower()]
            else:
                optimizer_cls = self.optimizer

            kwargs = self.optimizer_params.copy()
            kwargs['parameters'] = scaling
            return optimizer_cls(**kwargs)

        if self._fitted:
            raise RuntimeError("Already fitted scaling vector")

        self._fitted = True

        optimizer = get_optimizer()

        self.scaling_history.append(scaling.copy())
        self.scores_history.append(score())
        for epoch in tqdm.trange(self.n_epochs, disable=not self.show_progress):
            index = self.index_cls(**self.index_kwargs)
            index.fit(scaling * X_train)

            train_index = list(range(len(X_train)))
            if self.shuffle:
                self.random.shuffle(train_index)
            for idx in train_index:
                v = X_train[idx]
                y = y_train[idx]
                indices, _ = index.query_knn(v * scaling[0], batch_size)
                # skip `v` itself
                indices = indices[1:]
                nn_x = X_train[indices]
                nn_y = y_train[indices]

                cost, grad = self._get_gradient(scaling, nn_x, nn_y, v, y)

                if handle_possible_nans(grad):
                    self.scaling_ = scaling
                    return self.scaling_

                self.costs_history.append(cost)
                optimizer.step([grad])

            self.scaling_history.append(scaling.copy())
            self.scores_history.append(score())

        best_scaling = self.scaling_history[np.argmax(self.scores_history)]
        self.scaling_ = best_scaling
        return best_scaling

    def _get_gradient(
        self,
        scaling: np.ndarray,
        nn_x: np.ndarray,
        nn_y: np.ndarray,
        v: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the loss and the gradient.

        Args:
            scaling: The scaling vector.
            nn_x: The nearest neighbors of the current sample.
            nn_y: The targets of the nearest neighbors.
            v: The current sample.
            y: The target of the current sample.
        """
        q = nn_y - y
        delta = nn_x - v
        try:
            pinv = np.linalg.pinv(delta.T @ delta)
            nn_y_hat = pinv @ (delta.T @ q)
        except RuntimeError:
            raise RuntimeError(
                "Failed to compute psydo inverse!"
                f" The scaling vector was: {scaling}"
            )

        y_pred = y + delta @ nn_y_hat.T
        scaled_nn_x = nn_x * scaling
        scaled_v = v * scaling

        h_norm_in = scaled_nn_x - scaled_v
        h = np.clip(np.linalg.norm(h_norm_in, axis=1), self.epsilon, None)

        q = np.abs(nn_y - y_pred)

        vq = q - np.mean(q)
        vh = h - np.mean(h)

        if np.allclose(vq, 0) or np.allclose(vh, 0):
            # Either vq are all equal (this can be the case if the target value
            # is the same for all samples). Or vh are all equal to 0, which can
            # happen if the nearest neighbors are all the same. In any case, we
            # can't compute the cosine similarity in this case and therefore
            # return 0.
            return np.array([0.0]), np.zeros(scaling.shape[1])

        cossim = self._cossim(vq, vh)
        cost = -cossim
        # Backward path

        dcossim = -np.ones(1)  # ensure to account for - cossim
        _, dvh = self._cossim_backward(dcossim, cossim, vq, vh)

        # Derive: vh = h - np.mean(h)
        # d vh_j / d h_i =  - 1 / len(h)  if i != j
        # d vh_j / d h_i =  1 - 1 / len(h) if i == j
        #  -> I - 1/len(h)
        len_h = np.prod(h.shape)
        dim = dvh.shape[0]
        mean_len_matrix = np.full(dim, dim, 1 / len_h)
        mean_jac = np.eye(dim) - mean_len_matrix
        # dh = (1. - 1 / mean_len) * dvh
        dh = mean_jac @ dvh

        dh_norm_in = self._l2_norm_backward(dh, h, h_norm_in)

        # Derive: h_norm_in = scaled_nn_x - scaled_v
        dscaled_nn_x = dh_norm_in
        dscaled_v = -dh_norm_in

        # Derive: scaled_nn_x = nn_x * fsv
        dfsv_nn_x = nn_x * dscaled_nn_x
        # Derive: scaled_v = v * fsv
        dfsv_v = v * dscaled_v

        # Accumulate gradients
        dfsv = dfsv_nn_x + dfsv_v
        return cost, dfsv.sum(axis=0)

    @staticmethod
    def _l2_norm_backward(
        grad: np.ndarray, l2_norm: np.ndarray, a: np.ndarray
    ) -> np.ndarray:
        """Backward pass for the l2 norm.

        Args:
            grad: The backpropaged gradient.
            l2_norm: The l2 norm of the input.
            a: The input to the l2 norm.
        """
        # From: https://en.wikipedia.org/wiki/Norm_(mathematics)
        # d(||a||_2) / da = a / ||a||_2
        da = a / l2_norm[:, np.newaxis]
        return da * grad[:, np.newaxis]

    @staticmethod
    def _cossim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Computes the cosine similarity between two vectors."""
        return 1 - scipy.spatial.distance.cosine(a, b)

    @staticmethod
    def _cossim_backward(
        grad: np.ndarray,
        cossim: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        eps: float = 1e-8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Backward pass for the cosine similarity.

        Args:
            grad: The backpropaged gradient.
            cossim: The cosine similarity of the input.
            a: The first input to the cosine similarity.
            b: The second input to the cosine similarity.
            eps: The epsilon to avoid numerical issues.

        Returns:
            A tuple of the gradient of the first input and the gradient of the
            second input.
        """
        # From: https://math.stackexchange.com/questions/1923613/partial-derivative-of-cosine-similarity  # noqa
        #
        # d/da_i cossim(a, b) = b_i / (|a| |b|) - cossim(a, b) * a_i / |a|^2
        # analogously for b
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        dcos_da = (b / (na * nb + eps)) - (cossim * a / (na**2))
        dcos_db = (a / (na * nb + eps)) - (cossim * b / (nb**2))
        return dcos_da * grad, dcos_db * grad
