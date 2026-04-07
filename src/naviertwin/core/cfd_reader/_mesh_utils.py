"""CFD 리더 공통 메쉬 변환 유틸리티.

meshio Mesh 또는 PyVista DataSet → CFDDataset 변환 헬퍼를 제공한다.
세 리더(FluentReader, CGNSReader, GmshReader)가 공유한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from naviertwin.core.cfd_reader.base import CFDDataset
from naviertwin.utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def meshio_to_cfd_dataset(mesh: Any, source_file: str = "", reader: str = "") -> CFDDataset:
    """meshio Mesh 객체를 CFDDataset 으로 변환한다.

    Args:
        mesh: meshio.Mesh 인스턴스.
        source_file: 원본 파일 경로 (메타데이터용).
        reader: 리더 이름 (메타데이터용).

    Returns:
        변환된 CFDDataset.
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("pyvista 가 필요합니다: pip install pyvista") from exc

    ug: Any = pv.from_meshio(mesh)
    if not isinstance(ug, pv.UnstructuredGrid):
        ug = ug.cast_to_unstructured_grid()

    field_names = _collect_field_names(ug)
    logger.debug(
        "meshio → CFDDataset: n_points=%d, fields=%s", ug.n_points, field_names
    )

    return CFDDataset(
        mesh=ug,
        time_steps=[0.0],
        field_names=field_names,
        metadata={"reader": reader or "meshio", "source_file": source_file},
    )


def pyvista_to_cfd_dataset(mesh: Any, source_file: str = "", reader: str = "") -> CFDDataset:
    """PyVista DataSet 을 CFDDataset 으로 변환한다.

    Args:
        mesh: pyvista DataSet (UnstructuredGrid, MultiBlock 등).
        source_file: 원본 파일 경로 (메타데이터용).
        reader: 리더 이름 (메타데이터용).

    Returns:
        변환된 CFDDataset.
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("pyvista 가 필요합니다: pip install pyvista") from exc

    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine().cast_to_unstructured_grid()
    elif not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()

    field_names = _collect_field_names(mesh)
    logger.debug(
        "pyvista → CFDDataset: n_points=%d, fields=%s", mesh.n_points, field_names
    )

    return CFDDataset(
        mesh=mesh,
        time_steps=[0.0],
        field_names=field_names,
        metadata={"reader": reader or "pyvista", "source_file": source_file},
    )


def _collect_field_names(mesh: Any) -> list[str]:
    """point_data 와 cell_data 에서 필드 이름을 수집한다."""
    names: set[str] = set()
    if hasattr(mesh, "point_data"):
        names.update(str(k) for k in mesh.point_data.keys())
    if hasattr(mesh, "cell_data"):
        names.update(str(k) for k in mesh.cell_data.keys())
    return sorted(names)
