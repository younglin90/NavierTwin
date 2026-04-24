"""Stein Variational Gradient Descent (Liu & Wang 2016) — RBF kernel.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.uncertainty.svgd import svgd_step
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal((20, 1))
    >>> def grad_logp(x): return -x  # standard normal target
    >>> for _ in range(50):
    ...     x = svgd_step(x, grad_logp, lr=0.05)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def _rbf_kernel(
    X: NDArray[np.float64], h: float | None = None,
) -> tuple[NDArray, NDArray]:
    n, d = X.shape
    diff = X[:, None, :] - X[None, :, :]  # (n, n, d)
    sq = np.sum(diff ** 2, axis=-1)  # (n, n)
    if h is None:
        # median heuristic
        med = np.median(sq) if n > 1 else 1.0
        h = max(med / np.log(max(n, 2)), 1e-6)
    K = np.exp(-sq / (2 * h))
    grad_K = -diff / h * K[:, :, None]  # ∂K/∂x_i
    return K, grad_K


def svgd_step(
    X: NDArray[np.float64],
    grad_logp: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    lr: float = 0.01,
    h: float | None = None,
) -> NDArray[np.float64]:
    """1 SVGD update."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    K, gK = _rbf_kernel(X, h)
    glp = grad_logp(X)  # (n, d)
    phi = (K @ glp + gK.sum(axis=0)) / n
    return X + lr * phi


__all__ = ["svgd_step"]
