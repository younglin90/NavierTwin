"""Barycentric 좌표 + 삼각형 FE 보간.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.barycentric import barycentric_2d
    >>> tri = np.array([[0., 0.], [1., 0.], [0., 1.]])
    >>> bc = barycentric_2d(tri, np.array([0.25, 0.25]))
    >>> np.allclose(bc.sum(), 1.0)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required by barycentric analysis")


def barycentric_2d(
    triangle: NDArray[np.float64], p: NDArray[np.float64],
) -> NDArray[np.float64]:
    """삼각형 (3, 2) + 점 (2,) → barycentric coords (3,)."""
    return _kernels.barycentric_2d(np.asarray(triangle, dtype=np.float64), np.asarray(p, dtype=np.float64))


def triangle_interp(
    triangle: NDArray[np.float64], values: NDArray[np.float64],
    p: NDArray[np.float64],
) -> float:
    bc = barycentric_2d(triangle, p)
    return float(bc @ np.asarray(values))


def locate_triangle(
    points: NDArray[np.float64], simplices: NDArray[np.int64],
    p: NDArray[np.float64],
) -> int:
    """p 를 포함하는 삼각형 인덱스. 없으면 -1."""
    return int(
        _kernels.locate_triangle(
            np.asarray(points, dtype=np.float64),
            np.asarray(simplices, dtype=np.int64),
            np.asarray(p, dtype=np.float64),
        ),
    )


__all__ = ["barycentric_2d", "triangle_interp", "locate_triangle"]
