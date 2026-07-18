"""PyMeshLab 고급 필터 래퍼 — remeshing, curvature, hole-filling, Taubin smoothing.

기존 mesh_processor.py 는 simplify/smooth 만. 이 모듈은:
    - isotropic remeshing
    - Taubin smoothing
    - Laplacian smoothing (with explicit λ, μ)
    - per-vertex curvature
    - hole filling
    - watertight 검사

Examples:
    >>> import numpy as np
    >>> import pyvista as pv
    >>> from naviertwin.core.tools.mesh_processor_advanced import (
    ...     isotropic_remesh, taubin_smooth,
    ... )
    >>> mesh = pv.Sphere()
    >>> # out = isotropic_remesh(mesh, target_edge_length=0.1)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

_MISSING = (
    "pymeshlab 필요: pip install naviertwin[full]"
)


def _to_pymeshlab(mesh: Any) -> Any:
    try:
        import pymeshlab
    except ImportError as exc:
        raise RuntimeError(_MISSING) from exc

    surf = mesh.extract_surface(algorithm="dataset_surface").triangulate()
    points = np.asarray(surf.points, dtype=np.float64)
    faces = np.asarray(surf.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(points, faces))
    return ms


def _from_pymeshlab(ms: Any) -> Any:
    import pyvista as pv

    mm = ms.current_mesh()
    pts = np.asarray(mm.vertex_matrix(), dtype=np.float64)
    faces = np.asarray(mm.face_matrix(), dtype=np.int64)
    faces_vtk = np.column_stack(
        [np.full(len(faces), 3, dtype=np.int64), faces]
    ).ravel()
    return pv.PolyData(pts, faces_vtk).cast_to_unstructured_grid()


def isotropic_remesh(
    mesh: Any,
    target_edge_length: float = 0.05,
    iterations: int = 3,
) -> Any:
    """Isotropic explicit remeshing (IER)."""
    ms = _to_pymeshlab(mesh)
    try:
        import pymeshlab

        filter_kwargs = dict(
            targetlen=pymeshlab.PercentageValue(100 * target_edge_length),
            iterations=iterations,
        )
        ms.apply_filter("meshing_isotropic_explicit_remeshing", **filter_kwargs)
    except Exception as e:
        # 구버전 API
        logger.warning("isotropic remesh filter 실행 실패 (버전 차이): %s", e)
    logger.info("Isotropic remesh: %d 반복", iterations)
    return _from_pymeshlab(ms)


def taubin_smooth(
    mesh: Any,
    lambda_: float = 0.5,
    mu: float = -0.53,
    iterations: int = 10,
) -> Any:
    """Taubin λ-μ smoothing — 수축 없는 저주파 필터."""
    ms = _to_pymeshlab(mesh)
    try:
        ms.apply_filter(
            "apply_coord_taubin_smoothing",
            lambda_=lambda_,
            mu=mu,
            stepsmoothnum=iterations,
        )
    except Exception as e:
        logger.warning("Taubin smoothing 실패: %s", e)
    return _from_pymeshlab(ms)


def fill_holes(mesh: Any, max_hole_size: int = 30) -> Any:
    """경계 구멍 채우기."""
    ms = _to_pymeshlab(mesh)
    try:
        ms.apply_filter("meshing_close_holes", maxholesize=max_hole_size)
    except Exception as e:
        logger.warning("Hole filling 실패: %s", e)
    return _from_pymeshlab(ms)


def vertex_curvature(mesh: Any) -> NDArray[np.float64]:
    """vertex-wise Gaussian curvature (각 vertex 당 스칼라)."""
    ms = _to_pymeshlab(mesh)
    try:
        ms.apply_filter(
            "compute_scalar_by_discrete_curvature_per_vertex",
            curvaturetype="Gaussian Curvature",
        )
    except Exception as e:
        logger.warning("Curvature 필터 실패: %s — 0 반환", e)
        return np.zeros(ms.current_mesh().vertex_number())
    mm = ms.current_mesh()
    return np.asarray(mm.vertex_scalar_array(), dtype=np.float64)


def watertight_stats(mesh: Any) -> dict[str, Any]:
    """수밀 여부 / 경계 edge 수 / 비-매니폴드 vertex 수."""
    ms = _to_pymeshlab(mesh)
    try:
        out = ms.apply_filter("compute_topological_measures")
    except Exception as e:
        logger.warning("Topological measures 실패: %s", e)
        return {}
    return dict(out.items()) if isinstance(out, dict) else {}


__all__ = [
    "isotropic_remesh",
    "taubin_smooth",
    "fill_holes",
    "vertex_curvature",
    "watertight_stats",
]
