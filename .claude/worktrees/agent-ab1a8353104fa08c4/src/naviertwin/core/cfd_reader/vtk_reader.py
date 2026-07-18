"""VTK/VTU/VTP/PVD/STL/PLY 파일 리더 모듈.

PyVista 를 사용하여 다양한 VTK 계열 파일 포맷을 읽고
:class:`~naviertwin.core.cfd_reader.base.CFDDataset` 으로 변환한다.

단일 스냅샷 파일은 ``time_steps = [0.0]`` 으로 읽고,
PVD 컬렉션은 여러 VTU/VTP/VTK 스냅샷을 time-series 로 읽는다.

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
from xml.etree import ElementTree as ET

from naviertwin.core.cfd_reader.base import BaseReader, CFDDataset
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class VTKReader(BaseReader):
    """VTK/VTU/VTP/PVD/STL/PLY 파일 리더 (PyVista 기반).

    PyVista 가 지원하는 모든 VTK 계열 포맷을 읽는다.
    읽은 메쉬는 ``pv.UnstructuredGrid`` 로 변환된다.
    ``.pvd`` 파일은 동일 토폴로지 스냅샷 모음으로 해석한다.

    Attributes:
        supported_extensions: 지원하는 파일 확장자 집합.
    """

    supported_extensions: frozenset[str] = frozenset(
        {".vtk", ".vtu", ".vtp", ".pvd", ".stl", ".ply"}
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
        if suffix == ".pvd":
            return self._read_pvd(path)

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

    def _read_pvd(self, path: Path) -> CFDDataset:
        """PVD collection 파일을 time-series :class:`CFDDataset` 으로 읽는다."""
        try:
            import numpy as np
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(
                "PVD time-series 파일을 읽으려면 pyvista 와 numpy 가 필요합니다.\n"
                "  pip install pyvista numpy"
            ) from exc

        entries = self._parse_pvd_entries(path)
        if not entries:
            raise ValueError(f"PVD 파일에 DataSet 항목이 없습니다: {path}")

        meshes: list[Any] = []
        resolved_entries: list[tuple[float, Path]] = []
        entry_idx = 0
        while entry_idx < len(entries):
            time_value, file_name = entries[entry_idx]
            snapshot_path = (path.parent / file_name).resolve()
            if not snapshot_path.exists():
                raise FileNotFoundError(
                    f"PVD 스냅샷 파일이 존재하지 않습니다: {snapshot_path}"
                )
            raw_mesh = pv.read(str(snapshot_path))
            meshes.append(self._to_unstructured_grid(raw_mesh))
            resolved_entries.append((time_value, snapshot_path))
            entry_idx += 1

        meshes_with_entries = sorted(
            zip(resolved_entries, meshes, strict=True),
            key=lambda item: item[0][0],
        )
        resolved_entries = []
        meshes = []
        pair_idx = 0
        while pair_idx < len(meshes_with_entries):
            entry, mesh = meshes_with_entries[pair_idx]
            resolved_entries.append(entry)
            meshes.append(mesh)
            pair_idx += 1

        base_mesh = meshes[0].copy(deep=True)
        self._validate_same_topology(meshes, resolved_entries)

        field_locations = self._common_time_series_fields(meshes)
        time_series_fields: dict[str, Any] = {}
        field_items = list(field_locations.items())
        field_idx = 0
        while field_idx < len(field_items):
            field_name, location = field_items[field_idx]
            series = []
            mesh_idx = 0
            while mesh_idx < len(meshes):
                mesh = meshes[mesh_idx]
                container = mesh.point_data if location == "point" else mesh.cell_data
                series.append(np.asarray(container[field_name], dtype=float))
                mesh_idx += 1
            time_series_fields[field_name] = np.stack(series, axis=0)

            base_container = (
                base_mesh.point_data if location == "point" else base_mesh.cell_data
            )
            base_container[field_name] = series[0]
            field_idx += 1

        time_steps = []
        entry_idx = 0
        while entry_idx < len(resolved_entries):
            time_value, _ = resolved_entries[entry_idx]
            time_steps.append(float(time_value))
            entry_idx += 1
        field_names = sorted(field_locations)

        logger.debug(
            "PVD 읽기 완료: n_steps=%d, n_points=%d, n_cells=%d, fields=%s",
            len(time_steps),
            base_mesh.n_points,
            base_mesh.n_cells,
            field_names,
        )

        return CFDDataset(
            mesh=base_mesh,
            time_steps=time_steps,
            field_names=field_names,
            metadata={
                "reader": "VTKReader",
                "source_file": str(path),
                "source_files": self._resolved_entry_paths(resolved_entries),
                "time_series_fields": time_series_fields,
                "time_series_locations": dict(field_locations),
                "original_type": "PVDCollection",
            },
        )

    def _parse_pvd_entries(self, path: Path) -> list[tuple[float, str]]:
        """PVD XML에서 ``(timestep, file)`` 항목을 추출한다."""
        root = ET.parse(path).getroot()
        entries: list[tuple[float, str]] = []
        nodes = list(root.iter())
        index = 0
        while index < len(nodes):
            data_set = nodes[index]
            if self._strip_xml_namespace(data_set.tag) != "DataSet":
                index += 1
                continue
            file_name = data_set.attrib.get("file", "").strip()
            if not file_name:
                index += 1
                continue
            timestep_text = data_set.attrib.get("timestep")
            if timestep_text is None or timestep_text == "":
                timestep = float(index)
            else:
                try:
                    timestep = float(timestep_text)
                except ValueError as exc:
                    raise ValueError(
                        f"PVD timestep 값을 float 으로 해석할 수 없습니다: "
                        f"{timestep_text!r}"
                    ) from exc
            entries.append((timestep, file_name))
            index += 1
        return entries

    @staticmethod
    def _resolved_entry_paths(entries: list[tuple[float, Path]]) -> list[str]:
        paths: list[str] = []
        entry_idx = 0
        while entry_idx < len(entries):
            _, file_path = entries[entry_idx]
            paths.append(str(file_path))
            entry_idx += 1
        return paths

    @staticmethod
    def _strip_xml_namespace(tag: str) -> str:
        """``{namespace}Tag`` 형태의 XML 태그에서 로컬 이름만 반환한다."""
        return tag.rsplit("}", 1)[-1]

    @staticmethod
    def _validate_same_topology(
        meshes: list[Any],
        entries: list[tuple[float, Path]],
    ) -> None:
        """PVD time-series 스냅샷들이 동일 토폴로지인지 검증한다."""
        first = meshes[0]
        first_n_points = int(first.n_points)
        first_n_cells = int(first.n_cells)
        index = 1
        while index < len(meshes):
            mesh = meshes[index]
            same_points = int(mesh.n_points) == first_n_points
            same_cells = int(mesh.n_cells) == first_n_cells
            if not same_points or not same_cells:
                raise ValueError(
                    "PVD time-series 는 모든 스냅샷의 point/cell 개수가 같아야 "
                    f"합니다: {entries[index][1]}"
                )
            index += 1

    @staticmethod
    def _common_time_series_fields(meshes: list[Any]) -> dict[str, str]:
        """모든 스냅샷에 공통으로 존재하는 필드와 위치를 반환한다."""
        first = meshes[0]
        field_locations: dict[str, str] = {}
        point_names = list(first.point_data.keys())
        point_idx = 0
        while point_idx < len(point_names):
            field_locations[str(point_names[point_idx])] = "point"
            point_idx += 1

        cell_names = list(first.cell_data.keys())
        cell_idx = 0
        while cell_idx < len(cell_names):
            name = cell_names[cell_idx]
            field_locations.setdefault(str(name), "cell")
            cell_idx += 1

        mesh_idx = 1
        while mesh_idx < len(meshes):
            mesh = meshes[mesh_idx]
            retained: dict[str, str] = {}
            items = list(field_locations.items())
            item_idx = 0
            while item_idx < len(items):
                field_name, location = items[item_idx]
                container = mesh.point_data if location == "point" else mesh.cell_data
                if field_name in container:
                    retained[field_name] = location
                item_idx += 1
            field_locations = retained
            mesh_idx += 1
        return field_locations

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
            names.update(map(str, mesh.point_data.keys()))

        if hasattr(mesh, "cell_data"):
            names.update(map(str, mesh.cell_data.keys()))

        return sorted(names)
