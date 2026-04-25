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
    s_pos = 0.0
    s_neg = 0.0
    for i, v in enumerate(x):
        z = (v - mean) / sigma
        s_pos = max(0.0, s_pos + z - k)
        s_neg = min(0.0, s_neg + z + k)
        if s_pos > threshold or -s_neg > threshold:
            return i
    return -1


__all__ = ["cusum_detect"]
