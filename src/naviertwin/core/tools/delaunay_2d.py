"""2D Delaunay triangulation + P1 질량/강성 행렬 (scipy 기반).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.tools.delaunay_2d import triangulate
    >>> pts = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=float)
    >>> tri = triangulate(pts)
    >>> tri["simplices"].shape[1] == 3
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels


def triangulate(points: NDArray[np.float64]) -> dict:
    try:
        from scipy.spatial import Delaunay
    except ImportError as exc:
        raise RuntimeError("scipy 필요") from exc
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[1] != 2:
        raise ValueError("2D points 필요")
    tri = Delaunay(pts)
    return {"points": pts, "simplices": tri.simplices.astype(np.int64)}


def triangle_areas(
    points: NDArray[np.float64], simplices: NDArray[np.int64],
) -> NDArray[np.float64]:
    p = points[simplices]  # (n_tri, 3, 2)
    v1 = p[:, 1] - p[:, 0]
    v2 = p[:, 2] - p[:, 0]
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])


def lumped_mass_matrix(
    points: NDArray[np.float64], simplices: NDArray[np.int64],
) -> NDArray[np.float64]:
    """노드별 lumped mass = 인접 삼각형 면적 합 / 3."""
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by lumped_mass_matrix")
    return _kernels.lumped_mass_2d(
        np.asarray(points, dtype=np.float64),
        np.asarray(simplices, dtype=np.int64),
    )


def p1_stiffness_matrix(
    points: NDArray[np.float64], simplices: NDArray[np.int64],
) -> NDArray[np.float64]:
    """∇φ_i · ∇φ_j 적분 — P1 element. 반환 dense (n, n)."""
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by p1_stiffness_matrix")
    return _kernels.p1_stiffness_2d(
        np.asarray(points, dtype=np.float64),
        np.asarray(simplices, dtype=np.int64),
    )


__all__ = [
    "triangulate", "triangle_areas", "lumped_mass_matrix", "p1_stiffness_matrix",
]
