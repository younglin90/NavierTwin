"""Ordinary Kriging (scratch) — Gaussian variogram.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.surrogate.kriging_scratch import OrdinaryKriging
    >>> X = np.linspace(0, 1, 20)[:, None]
    >>> y = np.sin(X[:, 0] * 5)
    >>> k = OrdinaryKriging(theta=0.1, nugget=1e-6).fit(X, y)
    >>> yh = k.predict(np.array([[0.5]]))
    >>> abs(yh[0] - np.sin(2.5)) < 0.3
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _gaussian_cov(D: NDArray[np.float64], theta: float) -> NDArray[np.float64]:
    return np.exp(-(D / theta) ** 2)


class OrdinaryKriging:
    def __init__(self, theta: float = 1.0, nugget: float = 1e-6) -> None:
        self.theta = float(theta)
        self.nugget = float(nugget)
        self.X: NDArray | None = None
        self.mu: float = 0.0
        self._K_inv: NDArray | None = None
        self._centered: NDArray | None = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "OrdinaryKriging":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = X.shape[0]
        D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        K = _gaussian_cov(D, self.theta) + self.nugget * np.eye(n)
        self._K_inv = np.linalg.inv(K)
        one = np.ones(n)
        self.mu = float((one @ self._K_inv @ y) / (one @ self._K_inv @ one))
        self._centered = y - self.mu
        self.X = X
        return self

    def predict(
        self, Xq: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        assert self.X is not None and self._K_inv is not None
        Xq = np.atleast_2d(Xq)
        D = np.linalg.norm(Xq[:, None, :] - self.X[None, :, :], axis=2)
        kq = _gaussian_cov(D, self.theta)
        return self.mu + kq @ self._K_inv @ self._centered

    def predict_var(self, Xq: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.X is not None and self._K_inv is not None
        Xq = np.atleast_2d(Xq)
        D = np.linalg.norm(Xq[:, None, :] - self.X[None, :, :], axis=2)
        kq = _gaussian_cov(D, self.theta)
        var = 1.0 - np.einsum("qi,ij,qj->q", kq, self._K_inv, kq)
        return np.maximum(var, 0.0)


__all__ = ["OrdinaryKriging"]
