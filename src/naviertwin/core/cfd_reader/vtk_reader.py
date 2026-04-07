"""VTK/VTU/VTP/STL/PLY 파일 리더 모듈.

PyVista 를 사용하여 다양한 VTK 계열 파일 포맷을 읽고
:class:`~naviertwin.core.cfd_reader.base.CFDDataset` 으로 변환한다.

단일 스냅샷 파일만 지원하므로 ``time_steps = [0.0]`` 으로 고정된다.

Examples:
    직접 사용::

        from pathlib import Path
        from naviertwin.core.cfd_reader.vtk_reader import VTKReader

        reader = VTKReader()
        dataset = reader.read(Path("result.vtu"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from naviertwin.core.cfd_reader.base import BaseReader, CFDDataset
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class VTKReader(BaseReader):
    """VTK/VTU/VTP/STL/PLY 파일 리더 (PyVista 기반).

    PyVista 가 지원하는 모든 VTK 계열 포맷을 읽는다.
    읽은 메쉬는 ``pv.UnstructuredGrid`` 로 변환된다.

    Attributes:
        supported_extensions: 지원하는 파일 확장자 집합.
    """

    supported_extensions: frozenset[str] = frozenset(
        {".vtk", ".vtu", ".vtp", ".stl", ".ply"}
    )

    # ------------------------------------------------------------------
    # BaseReader 인터페이스
    # ------------------------------------------------------------------

    def read(self, path: Path) -> CFDDataset:
        """VTK 계열 파일을 읽어 :class:`CFDDataset` 으로 반환한다.

        Args:
            path: 읽을 VTK 파일 경로.

        Returns:
            파싱된 :class:`CFDDataset`. ``time_steps = [0.0]`` 고정.

        Raises:
            FileNotFoundError: 경로가 존재하지 않는 경우.
            ImportError: pyvista 가 설치되어 있지 않은 경우.
            ValueError: 지원하지 않는 파일 포맷인 경우.
        """
        if not path.exists():
            raise FileNotFoundError(f"파일이 존재하지 않습니다: {path}")

        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(
                "VTK 파일을 읽으려면 pyvista 가 필요합니다.\n"
                "  pip install pyvista"
            ) from exc

        suffix = path.suffix.lower()
        if suffix not in self.supported_extensions:
            raise ValueError(
                f"지원하지 않는 확장자입니다: '{suffix}'\n"
                f"지원 확장자: {sorted(self.supported_extensions)}"
            )

        logger.info("VTK 파일 읽기: %s", path)
        raw_mesh = pv.read(str(path))

        ug = self._to_unstructured_grid(raw_mesh)
        field_names = self._extract_field_names(ug)

        logger.debug(
            "읽기 완료: n_points=%d, n_cells=%d, fields=%s",
            ug.n_points,
            ug.n_cells,
            field_names,
        )

        return CFDDataset(
            mesh=ug,
            time_steps=[0.0],
            field_names=field_names,
            metadata={
                "reader": "VTKReader",
                "source_file": str(path),
                "original_type": type(raw_mesh).__name__,
            },
        )

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _to_unstructured_grid(self, mesh: Any) -> Any:
        """PyVista 메쉬를 UnstructuredGrid 로 변환한다.

        이미 UnstructuredGrid 이면 그대로 반환하고,
        그 외 타입은 ``cast_to_unstructured_grid()`` 를 시도한다.
        MultiBlock 이면 먼저 ``combine()`` 으로 합친다.

        Args:
            mesh: 변환할 PyVista DataSet 또는 MultiBlock.

        Returns:
            ``pv.UnstructuredGrid`` 인스턴스.
        """
        try:
            import pyvista as pv
        except ImportError:
            return mesh

        if isinstance(mesh, pv.UnstructuredGrid):
            return mesh

        if isinstance(mesh, pv.MultiBlock):
            logger.debug("MultiBlock → combine → UnstructuredGrid")
            combined = mesh.combine()
            return combined.cast_to_unstructured_grid()

        logger.debug(
            "%s → cast_to_unstructured_grid", type(mesh).__name__
        )
        return mesh.cast_to_unstructured_grid()

    def _extract_field_names(self, mesh: Any) -> list[str]:
        """메쉬의 point_data 와 cell_data 에서 필드 이름을 수집한다.

        Args:
            mesh: PyVista DataSet 객체.

        Returns:
            중복 없는 정렬된 필드 이름 리스트.
        """
        names: set[str] = set()

        if hasattr(mesh, "point_data"):
            names.update(str(k) for k in mesh.point_data.keys())

        if hasattr(mesh, "cell_data"):
            names.update(str(k) for k in mesh.cell_data.keys())

        return sorted(names)
