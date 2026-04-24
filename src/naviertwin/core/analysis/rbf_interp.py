"""RBF 보간 — Gaussian / multiquadric / thin-plate spline.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.rbf_interp import RBFInterpolator
    >>> X = np.array([[0.], [1.], [2.], [3.]])
    >>> y = np.array([0., 1., 4., 9.])
    >>> rbf = RBFInterpolator(X, y, kernel="multiquadric", epsilon=1.0)
    >>> abs(float(rbf(np.array([[1.5]]))[0]) - 2.25) < 0.5
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _phi(r: NDArray, kernel: str, eps: float) -> NDArray:
    if kernel == "gaussian":
        return np.exp(-(eps * r) ** 2)
    if kernel == "multiquadric":
        return np.sqrt(1.0 + (eps * r) ** 2)
    if kernel == "inverse_multiquadric":
        return 1.0 / np.sqrt(1.0 + (eps * r) ** 2)
    if kernel == "thin_plate":
        with np.errstate(divide="ignore"):
            val = np.where(r > 0, r ** 2 * np.log(r), 0.0)
        return val
    raise ValueError(f"unknown kernel: {kernel}")


class RBFInterpolator:
    """fit on centers X → predict at new points."""

    def __init__(
        self, X: NDArray[np.float64], y: NDArray[np.float64],
        *, kernel: str = "gaussian", epsilon: float = 1.0,
        reg: float = 0.0,
    ) -> None:
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64).ravel()
        self.kernel = kernel
        self.eps = float(epsilon)
        n = self.X.shape[0]
        # center-center distance matrix
        D = np.linalg.norm(self.X[:, None, :] - self.X[None, :, :], axis=2)
        A = _phi(D, kernel, self.eps)
        if reg > 0:
            A = A + reg * np.eye(n)
        self.w = np.linalg.solve(A, self.y)

    def __call__(self, Xq: NDArray[np.float64]) -> NDArray[np.float64]:
        Xq = np.asarray(Xq, dtype=np.float64)
        if Xq.ndim == 1:
            Xq = Xq[None, :]
        D = np.linalg.norm(Xq[:, None, :] - self.X[None, :, :], axis=2)
        return _phi(D, self.kernel, self.eps) @ self.w


__all__ = ["RBFInterpolator"]
