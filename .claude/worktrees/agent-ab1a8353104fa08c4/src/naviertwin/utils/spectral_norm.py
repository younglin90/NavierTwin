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
    """Largest singular value via native power iteration on W.T @ W."""
    from naviertwin.core.linalg.power_method import power_iteration

    W = np.asarray(W, dtype=np.float64)
    gram = W.T @ W
    lam, _ = power_iteration(gram, n_iter=int(n_iter), seed=0)
    return float(np.sqrt(max(lam, 0.0)))


def sn_penalty(W: NDArray[np.float64], *, target: float = 1.0) -> float:
    s = spectral_norm(W)
    return float(max(0.0, s - target) ** 2)


__all__ = ["sn_penalty", "spectral_norm"]
