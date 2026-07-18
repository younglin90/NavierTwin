"""관심영역 마스크 — box / sphere / cylinder / plane / predicate.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.roi_mask import box_mask, sphere_mask
    >>> pts = np.array([[0,0,0],[1,1,1],[5,5,5]], dtype=float)
    >>> box_mask(pts, (-0.5,-0.5,-0.5,1.5,1.5,1.5)).tolist()
    [True, True, False]
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def box_mask(
    points: NDArray[np.float64],
    box: tuple[float, float, float, float, float, float],
) -> NDArray[np.bool_]:
    """(xmin, ymin, zmin, xmax, ymax, zmax) 안쪽."""
    p = np.asarray(points, dtype=np.float64)
    xmin, ymin, zmin, xmax, ymax, zmax = box
    return (
        (p[:, 0] >= xmin) & (p[:, 0] <= xmax)
        & (p[:, 1] >= ymin) & (p[:, 1] <= ymax)
        & (p[:, 2] >= zmin) & (p[:, 2] <= zmax)
    )


def sphere_mask(
    points: NDArray[np.float64],
    center: tuple[float, float, float],
    radius: float,
) -> NDArray[np.bool_]:
    p = np.asarray(points, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    d = np.linalg.norm(p - c, axis=1)
    return d <= radius


def cylinder_mask(
    points: NDArray[np.float64],
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
) -> NDArray[np.bool_]:
    p = np.asarray(points, dtype=np.float64)
    s = np.asarray(start, dtype=np.float64)
    e = np.asarray(end, dtype=np.float64)
    axis = e - s
    L = np.linalg.norm(axis)
    if L <= 0:
        raise ValueError("invalid axis length")
    axis_u = axis / L
    rel = p - s
    # 투영 거리
    t = rel @ axis_u
    # 축에서 radial 거리
    perp = rel - np.outer(t, axis_u)
    radial = np.linalg.norm(perp, axis=1)
    return (t >= 0) & (t <= L) & (radial <= radius)


def plane_half_space(
    points: NDArray[np.float64],
    origin: tuple[float, float, float],
    normal: tuple[float, float, float],
) -> NDArray[np.bool_]:
    """점이 평면의 정방향 쪽인지."""
    p = np.asarray(points, dtype=np.float64)
    o = np.asarray(origin, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    return (p - o) @ n >= 0


def predicate_mask(
    points: NDArray[np.float64], fn: Callable[[NDArray], NDArray],
) -> NDArray[np.bool_]:
    return fn(np.asarray(points, dtype=np.float64)).astype(bool)


__all__ = [
    "box_mask", "sphere_mask", "cylinder_mask",
    "plane_half_space", "predicate_mask",
]
