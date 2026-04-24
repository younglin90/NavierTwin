"""Mesh clip plane — half-space culling.

n · (x - p) > 0 인 점만 유지. 삼각형 분할은 단순 vertex-mask 기반.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.tools.clip_plane import clip_points
    >>> pts = np.array([[1., 0, 0], [-1., 0, 0]])
    >>> mask = clip_points(pts, n=np.array([1.,0,0]), p=np.zeros(3))
    >>> mask.tolist()
    [True, False]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def clip_points(
    points: NDArray[np.float64],
    n: NDArray[np.float64],
    p: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """평면 (n, p) 의 + 쪽 (n·(x-p) > 0) 점만 True."""
    pts = np.asarray(points, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return (pts - p) @ n > 0


def clip_triangles(
    points: NDArray[np.float64],
    triangles: NDArray[np.int_],
    n: NDArray[np.float64],
    p: NDArray[np.float64],
) -> NDArray[np.int_]:
    """모든 vertex 가 + 쪽인 삼각형만 유지."""
    mask_v = clip_points(points, n, p)
    tri = np.asarray(triangles)
    keep = mask_v[tri].all(axis=1)
    return tri[keep]


__all__ = ["clip_points", "clip_triangles"]
