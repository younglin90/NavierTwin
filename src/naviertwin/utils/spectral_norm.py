"""Spectral-norm computation + regularization penalty (numpy-based).

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.spectral_norm import spectral_norm, sn_penalty
    >>> W = np.array([[1.0, 0], [0, 2.0]])
    >>> spectral_norm(W)
    2.0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def spectral_norm(W: NDArray[np.float64], *, n_iter: int = 30) -> float:
    """Power iteration on W."""
    W = np.asarray(W, dtype=np.float64)
    rng = np.random.default_rng(0)
    v = rng.standard_normal(W.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(n_iter):
        u = W @ v
        u /= np.linalg.norm(u) + 1e-30
        v = W.T @ u
        nv = np.linalg.norm(v)
        v /= nv + 1e-30
    return float(np.linalg.norm(W @ v))


def sn_penalty(W: NDArray[np.float64], *, target: float = 1.0) -> float:
    s = spectral_norm(W)
    return float(max(0.0, s - target) ** 2)


__all__ = ["sn_penalty", "spectral_norm"]
