"""Marching cubes-lite — 간단 2D version (marching squares) + 3D voxel iso-segments.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.tools.marching_cubes_lite import marching_squares
    >>> f = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    >>> segs = marching_squares(f, level=0.5)
    >>> len(segs) > 0
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels


def marching_squares(
    f: NDArray[np.float64], level: float = 0.0,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """2D iso-contour 추출. (i, j) 인덱스 좌표 반환."""
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by marching_squares")
    return _kernels.marching_squares(np.asarray(f, dtype=np.float64), level)


__all__ = ["marching_squares"]
