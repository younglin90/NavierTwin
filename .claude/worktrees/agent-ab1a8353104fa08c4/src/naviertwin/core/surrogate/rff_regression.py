"""Gaussian Random Fourier Features 기반 linear regression (RBF kernel approx).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.surrogate.rff_regression import RFFRegression
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(-1, 1, size=(50, 1))
    >>> y = np.sin(3 * X[:, 0])
    >>> r = RFFRegression(num_features=128, sigma=0.5, ridge=1e-3, seed=0).fit(X, y)
    >>> y_hat = r.predict(X)
    >>> np.corrcoef(y_hat, y)[0, 1] > 0.95
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.core.neural.positional_enc import gaussian_rff

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


class RFFRegression:
    def __init__(
        self, num_features: int = 256, sigma: float = 1.0,
        ridge: float = 1e-4, seed: int | None = 0,
    ) -> None:
        self.num_features = int(num_features)
        self.sigma = float(sigma)
        self.ridge = float(ridge)
        self.seed = seed
        self.W_out: NDArray | None = None

    def _features(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return gaussian_rff(
            X, num_features=self.num_features, sigma=self.sigma, seed=self.seed,
        )

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "RFFRegression":
        Phi = self._features(X)
        A = Phi.T @ Phi + self.ridge * np.eye(Phi.shape[1])
        self.W_out = np.asarray(_kernels.solve_square(A, Phi.T @ np.asarray(y).ravel()), dtype=np.float64)
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.W_out is None:
            raise RuntimeError("fit 먼저")
        return self._features(X) @ self.W_out


__all__ = ["RFFRegression"]
