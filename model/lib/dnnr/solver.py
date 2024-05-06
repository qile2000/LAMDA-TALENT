from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse.linalg import lsqr
from sklearn.linear_model import Lasso, LinearRegression, Ridge


class Solver(ABC):
    @abstractmethod
    def solve(self, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Returns the solution of a linear equation.

        Finds a solution to the following equation:

            a^T X = b

        and `w` is a weighting of each datapoint in X.

        Args:
            a: with shape (n_samples, n_features)
            b: with shape (n_samples, 1)

        Returns:
            The solution of the equation.
        """


class SKLinearRegression(Solver):
    def __init__(self):
        self.solver = LinearRegression(fit_intercept=False)

    def solve(self, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
        self.solver.fit(a, b, sample_weight=w)
        return self.solver.coef_


class SKRidge(Solver):
    def __init__(self):
        self.solver = Ridge(fit_intercept=False)

    def solve(self, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
        self.solver.fit(a, b, sample_weight=w)
        return self.solver.coef_


class SKLasso(Solver):
    def __init__(self):
        self.solver = Lasso(fit_intercept=False)

    def solve(self, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
        self.solver.fit(a, b, sample_weight=w)
        return self.solver.coef_


class ScipyLsqr(Solver):
    def solve(self, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
        C = np.diag(w)
        X = a.T @ C @ a
        Y = a.T @ C @ b
        return lsqr(X, Y)[0]


class NPSolver(Solver):
    def solve(self, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
        C = np.diag(w)
        w_wlr = np.linalg.pinv(a.T @ C @ a) @ (a.T @ C @ b)
        return w_wlr


def create_solver(solver: str) -> Solver:
    if solver == "linear_regression":
        return SKLinearRegression()
    elif solver == "scipy_lsqr":
        return ScipyLsqr()
    elif solver == "numpy":
        return NPSolver()
    elif solver == "ridge":
        return SKRidge()
    elif solver == "lasso":
        return SKLasso()
    else:
        raise NotImplementedError(
            "Solver method not implemented or not recognized"
        )
