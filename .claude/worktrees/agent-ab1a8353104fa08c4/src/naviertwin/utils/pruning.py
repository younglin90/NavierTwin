"""Magnitude pruning — zero out smallest |w| fraction.

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.pruning import prune_magnitude
    >>> w = np.array([1.0, -0.1, 2.0, 0.05])
    >>> wp = prune_magnitude(w, sparsity=0.5)
    >>> (wp == 0).sum()
    2
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def prune_magnitude(
    weights: NDArray[np.float64], *, sparsity: float = 0.5,
) -> NDArray[np.float64]:
    """Zero the smallest |w| `sparsity` fraction."""
    w = np.asarray(weights, dtype=np.float64).copy()
    flat = np.abs(w).ravel()
    if flat.size == 0:
        return w
    k = int(round(sparsity * flat.size))
    if k <= 0:
        return w
    thresh = np.partition(flat, k - 1)[k - 1]
    w[np.abs(w) <= thresh] = 0.0
    return w


__all__ = ["prune_magnitude"]
