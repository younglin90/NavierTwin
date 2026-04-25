"""3D quiver downsample — stride sampling.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.visualization.quiver_downsample import downsample_3d
    >>> X, Y, Z = np.mgrid[:10, :10, :10]
    >>> U = np.zeros_like(X, dtype=float)
    >>> pts, vecs = downsample_3d(np.stack([X, Y, Z], -1), np.stack([U, U, U], -1), stride=2)
    >>> pts.shape[0]
    125
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def downsample_3d(
    points: NDArray[np.float64],
    vectors: NDArray[np.float64],
    *,
    stride: int = 2,
) -> tuple[NDArray, NDArray]:
    """points/vectors shape (nx, ny, nz, 3) → flattened with stride."""
    p = np.asarray(points)
    v = np.asarray(vectors)
    p_ds = p[::stride, ::stride, ::stride].reshape(-1, 3)
    v_ds = v[::stride, ::stride, ::stride].reshape(-1, 3)
    return p_ds, v_ds


__all__ = ["downsample_3d"]
