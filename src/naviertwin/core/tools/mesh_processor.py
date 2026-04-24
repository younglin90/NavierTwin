"""메쉬 후처리(단순화·스무딩·품질 검사) 모듈.

- ``simplify`` / ``smooth`` : PyMeshLab 필요 (optional, ``naviertwin[full]``)
- ``quality_report`` : PyVista 만으로 동작 — core 의존성만으로 가능

Examples:
    품질 보고::

        from naviertwin.core.tools.mesh_processor import quality_report
        rep = quality_report(mesh)
        print(rep["aspect_ratio_mean"])
"""

from __future__ import annotations

from typing import Any

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

_PYMESHLAB_MISSING = (
    "pymeshlab 설치 필요: pip install naviertwin[full]\n"
    "또는 직접 설치: pip install pymeshlab"
)


def _to_pymeshlab_mesh(mesh: Any) -> Any:
    """PyVista UnstructuredGrid → pymeshlab MeshSet 변환."""
    try:
        import numpy as np
        import pymeshlab
    except ImportError as exc:
        raise RuntimeError(_PYMESHLAB_MISSING) from exc

    surf = mesh.extract_surface(algorithm="dataset_surface").triangulate()
    points = np.asarray(surf.points, dtype=np.float64)
    faces = np.asarray(surf.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(points, faces))
    return ms


def _from_pymeshlab_mesh(ms: Any) -> Any:
    """pymeshlab MeshSet → PyVista PolyData → UnstructuredGrid 변환."""
    import numpy as np
    import pyvista as pv

    mm = ms.current_mesh()
    pts = np.asarray(mm.vertex_matrix(), dtype=np.float64)
    faces = np.asarray(mm.face_matrix(), dtype=np.int64)
    faces_vtk = np.column_stack(
        [np.full(len(faces), 3, dtype=np.int64), faces]
    ).ravel()
    poly = pv.PolyData(pts, faces_vtk)
    return poly.cast_to_unstructured_grid()


def simplify(mesh: Any, target_faces: int = 1000) -> Any:
    """메쉬를 목표 면 수까지 단순화한다.

    PyMeshLab 의 Quadric Edge Collapse Decimation 을 사용한다.

    Args:
        mesh: PyVista DataSet.
        target_faces: 목표 면 수.

    Returns:
        단순화된 PyVista UnstructuredGrid.

    Raises:
        RuntimeError: pymeshlab 이 설치되지 않은 경우.
    """
    ms = _to_pymeshlab_mesh(mesh)
    ms.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=int(target_faces),
    )
    logger.info(
        "메쉬 단순화: 목표 %d 면 → 결과 %d 면",
        target_faces,
        ms.current_mesh().face_number(),
    )
    return _from_pymeshlab_mesh(ms)


def smooth(mesh: Any, iterations: int = 10, lambda_: float = 0.5) -> Any:
    """Laplacian 스무딩을 적용한다.

    Args:
        mesh: PyVista DataSet.
        iterations: 스무딩 반복 횟수.
        lambda_: 스무딩 강도 [0, 1].

    Returns:
        스무딩된 PyVista UnstructuredGrid.

    Raises:
        RuntimeError: pymeshlab 이 설치되지 않은 경우.
    """
    ms = _to_pymeshlab_mesh(mesh)
    ms.apply_filter(
        "apply_coord_laplacian_smoothing",
        stepsmoothnum=int(iterations),
    )
    logger.info("메쉬 스무딩: iterations=%d, lambda=%.2f", iterations, lambda_)
    return _from_pymeshlab_mesh(ms)


def _compute_cell_quality(mesh: Any, measure: str) -> Any:
    """PyVista 버전 차이를 흡수하여 셀 품질을 계산한다."""
    if hasattr(mesh, "cell_quality"):
        return mesh.cell_quality(measure)
    return mesh.compute_cell_quality(quality_measure=measure)


def quality_report(mesh: Any) -> dict[str, float]:
    """PyVista 기반 셀 품질 보고서를 반환한다.

    pymeshlab 이 없어도 동작한다.

    Args:
        mesh: PyVista DataSet.

    Returns:
        키: aspect_ratio_{mean,min,max}, scaled_jacobian_{mean,min},
        n_points, n_cells, volume(3D)/area(2D).
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy 가 필요합니다") from exc

    report: dict[str, float] = {
        "n_points": float(mesh.n_points),
        "n_cells": float(mesh.n_cells),
    }

    try:
        qual = _compute_cell_quality(mesh, "aspect_ratio")
        ar = np.asarray(qual.cell_data.get("aspect_ratio", qual.cell_data.get("CellQuality", [])))
        ar = ar[np.isfinite(ar)]
        if ar.size > 0:
            report["aspect_ratio_mean"] = float(np.mean(ar))
            report["aspect_ratio_min"] = float(np.min(ar))
            report["aspect_ratio_max"] = float(np.max(ar))
    except Exception as e:  # noqa: BLE001
        logger.debug("aspect_ratio 계산 실패: %s", e)

    try:
        qual = _compute_cell_quality(mesh, "scaled_jacobian")
        sj = np.asarray(qual.cell_data.get("scaled_jacobian", qual.cell_data.get("CellQuality", [])))
        sj = sj[np.isfinite(sj)]
        if sj.size > 0:
            report["scaled_jacobian_mean"] = float(np.mean(sj))
            report["scaled_jacobian_min"] = float(np.min(sj))
    except Exception as e:  # noqa: BLE001
        logger.debug("scaled_jacobian 계산 실패: %s", e)

    try:
        sizes = mesh.compute_cell_sizes(length=False, area=True, volume=True)
        vol = np.asarray(sizes.cell_data.get("Volume", [0.0]))
        area = np.asarray(sizes.cell_data.get("Area", [0.0]))
        total_vol = float(np.sum(vol))
        total_area = float(np.sum(area))
        if total_vol > 0:
            report["volume"] = total_vol
        if total_area > 0:
            report["area"] = total_area
    except Exception as e:  # noqa: BLE001
        logger.debug("size 계산 실패: %s", e)

    return report


__all__ = ["simplify", "smooth", "quality_report"]
