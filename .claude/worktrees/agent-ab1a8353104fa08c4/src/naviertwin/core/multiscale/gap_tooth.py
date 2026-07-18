"""Gap-tooth (patch dynamics) — only simulate small patches, interpolate gaps.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.multiscale.gap_tooth import gap_tooth_fill
    >>> patch_centers = np.array([2, 5, 8])
    >>> patch_vals = np.array([1.0, 2.0, 3.0])
    >>> u_full = gap_tooth_fill(patch_centers, patch_vals, n_full=10)
    >>> u_full.shape
    (10,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def gap_tooth_fill(
    patch_centers: NDArray[np.int_],
    patch_vals: NDArray[np.float64],
    *,
    n_full: int,
) -> NDArray[np.float64]:
    """Linear interp between patch centers to fill grid."""
    x = np.arange(n_full)
    return np.interp(x, np.asarray(patch_centers), np.asarray(patch_vals))


__all__ = ["gap_tooth_fill"]
