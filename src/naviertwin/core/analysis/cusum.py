"""CUSUM change-point detection — Page 1954.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.cusum import cusum_detect
    >>> x = np.concatenate([np.zeros(50), np.ones(50)])
    >>> idx = cusum_detect(x, threshold=3.0)
    >>> idx
    50
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def cusum_detect(
    x: NDArray[np.float64], *,
    threshold: float = 3.0, mean: float | None = None,
    sigma: float | None = None, k: float = 0.5,
) -> int:
    """Page CUSUM with reference value k (in units of σ).

    s_pos[i] = max(0, s_pos[i-1] + (x_i - mean)/σ - k)
    s_neg[i] = min(0, s_neg[i-1] + (x_i - mean)/σ + k)
    Returns first index where |S| > threshold; -1 if none.
    """
    x = np.asarray(x, dtype=np.float64)
    if mean is None:
        mean = float(x[:max(10, len(x) // 4)].mean())
    if sigma is None:
        sigma = float(x[:max(10, len(x) // 4)].std() + 1e-12)
    return int(_kernels.cusum_detect(x, float(threshold), float(mean), float(sigma), float(k)))


__all__ = ["cusum_detect"]
