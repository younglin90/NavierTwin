"""Takens delay embedding + mutual information 지연 추정.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.system_id.delay_embed import delay_embed
    >>> x = np.sin(np.linspace(0, 10, 500))
    >>> Y = delay_embed(x, dim=3, delay=5)
    >>> Y.shape
    (490, 3)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels


def delay_embed(
    x: NDArray[np.float64], dim: int = 3, delay: int = 1,
) -> NDArray[np.float64]:
    """x (T,) → (T - (dim-1)*delay, dim)."""
    if _kernels is None:
        raise ImportError("naviertwin._native._kernels is required by delay_embed")
    x = np.asarray(x, dtype=np.float64).ravel()
    m = x.size - (dim - 1) * delay
    if m <= 0:
        raise ValueError("너무 짧은 시계열")
    return _kernels.delay_embed_1d(x, dim, delay)


def autocorrelation(x: NDArray[np.float64], max_lag: int = 50) -> NDArray[np.float64]:
    if _kernels is None:
        raise ImportError("naviertwin._native._kernels is required by autocorrelation")
    x = np.asarray(x, dtype=np.float64).ravel()
    return _kernels.autocorrelation_1d(x, max_lag)


def first_zero_crossing(corr: NDArray[np.float64]) -> int:
    """autocorr 첫 zero-crossing → 추천 delay."""
    i = 1
    while i < corr.size:
        if corr[i] <= 0:
            return i
        i += 1
    return corr.size - 1


__all__ = ["delay_embed", "autocorrelation", "first_zero_crossing"]
