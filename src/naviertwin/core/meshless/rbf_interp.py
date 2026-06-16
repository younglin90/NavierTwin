"""RBF (radial basis function) interpolation — Gaussian/multiquadric.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.meshless.rbf_interp import RBFInterp
    >>> X = np.array([[0.], [1.], [2.]])
    >>> y = np.array([0.0, 1.0, 4.0])
    >>> r = RBFInterp(eps=1.0).fit(X, y)
    >>> abs(r.predict(np.array([[1.]])).item() - 1.0) < 1e-6
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def _gaussian(r2, eps):
    return np.exp(-eps * r2)


class RBFInterp:
    def __init__(self, eps: float = 1.0) -> None:
        self.eps = eps
        self.X: NDArray | None = None
        self.w: NDArray | None = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "RBFInterp":
        self.X = np.asarray(X, dtype=np.float64)
        d = self.X[:, None, :] - self.X[None, :, :]
        K = _gaussian(np.sum(d ** 2, axis=-1), self.eps)
        self.w = np.asarray(_kernels.solve_square(K + 1e-10 * np.eye(K.shape[0]), np.asarray(y)), dtype=np.float64)
        return self

    def predict(self, X_q: NDArray[np.float64]) -> NDArray[np.float64]:
        Xq = np.asarray(X_q, dtype=np.float64)
        d = Xq[:, None, :] - self.X[None, :, :]
        K = _gaussian(np.sum(d ** 2, axis=-1), self.eps)
        return K @ self.w


__all__ = ["RBFInterp"]
