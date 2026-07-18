"""Signed Distance Function — primitive shapes (sphere, box, plane).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.geometry.sdf import sdf_sphere, sdf_box
    >>> sdf_sphere(np.array([0., 0, 0]), center=np.zeros(3), r=1.0)
    -1.0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sdf_sphere(p: NDArray, *, center: NDArray, r: float) -> float | NDArray:
    p = np.asarray(p, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    return np.linalg.norm(p - c, axis=-1) - r


def sdf_box(p: NDArray, *, center: NDArray, half_size: NDArray) -> float | NDArray:
    p = np.asarray(p, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    h = np.asarray(half_size, dtype=np.float64)
    q = np.abs(p - c) - h
    out = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(np.max(q, axis=-1), 0)
    return out


def sdf_plane(p: NDArray, *, normal: NDArray, offset: float = 0.0) -> float | NDArray:
    n = np.asarray(normal) / (np.linalg.norm(normal) + 1e-30)
    return np.asarray(p) @ n - offset


__all__ = ["sdf_box", "sdf_plane", "sdf_sphere"]
