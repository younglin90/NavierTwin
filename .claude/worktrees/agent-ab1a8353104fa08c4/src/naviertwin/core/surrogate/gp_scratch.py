"""Gaussian Process — RBF kernel + LML + optim.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.surrogate.gp_scratch import GPRegressor
    >>> X = np.linspace(0, 1, 20)[:, None]
    >>> y = np.sin(X[:, 0] * 5) + 0.05 * np.random.default_rng(0).standard_normal(20)
    >>> gp = GPRegressor(lengthscale=0.1, sigma=1.0).fit(X, y)
    >>> mu, var = gp.predict(np.array([[0.5]]))
    >>> abs(mu[0] - np.sin(2.5)) < 0.5
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def rbf_kernel(
    X: NDArray[np.float64], Y: NDArray[np.float64],
    lengthscale: float, sigma: float,
) -> NDArray[np.float64]:
    D = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
    return sigma ** 2 * np.exp(-0.5 * (D / lengthscale) ** 2)


class GPRegressor:
    def __init__(
        self, lengthscale: float = 1.0, sigma: float = 1.0, noise: float = 1e-4,
    ) -> None:
        self.l = float(lengthscale)
        self.s = float(sigma)
        self.noise = float(noise)
        self.X: NDArray | None = None
        self.y: NDArray | None = None
        self._L: NDArray | None = None
        self._alpha: NDArray | None = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "GPRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = X.shape[0]
        K = rbf_kernel(X, X, self.l, self.s) + self.noise * np.eye(n)
        self._L = np.linalg.cholesky(K)
        z = np.asarray(_kernels.solve_square(self._L, y), dtype=np.float64)
        self._alpha = np.asarray(_kernels.solve_square(self._L.T, z), dtype=np.float64)
        self.X = X
        self.y = y
        return self

    def predict(
        self, Xq: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        assert self.X is not None and self._L is not None
        Ks = rbf_kernel(Xq, self.X, self.l, self.s)
        mu = Ks @ self._alpha
        v = np.asarray(_kernels.solve_square(self._L, Ks.T), dtype=np.float64)
        Kss = rbf_kernel(Xq, Xq, self.l, self.s)
        var = np.diag(Kss - v.T @ v)
        return mu, np.clip(var, 0.0, None)

    def log_marginal_likelihood(self) -> float:
        assert self.X is not None and self._L is not None
        n = self.X.shape[0]
        logdet = 2.0 * np.sum(np.log(np.diag(self._L)))
        term1 = -0.5 * self.y @ self._alpha
        term2 = -0.5 * logdet
        term3 = -0.5 * n * np.log(2 * np.pi)
        return float(term1 + term2 + term3)


__all__ = ["GPRegressor", "rbf_kernel"]
