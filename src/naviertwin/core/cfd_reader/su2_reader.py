"""SU2 .su2 파일 리더 모듈.

SU2 네이티브 ASCII 메쉬 포맷(.su2)과 sibling .csv 결과 파일을 읽는다.

폴백 체인:
    1. meshio (SU2 공식 지원, IO 표준)
    2. SU2ASCIIParser (NDIME / NPOIN / NELEM / NMARK 섹션 직접 파싱)

.csv 사이드카 처리:
    .su2 경로와 같은 디렉토리(혹은 동일한 베이스네임)의 .csv / _surface.csv /
    restart.csv / history.csv 파일을 자동 감지하여 point_data 로 주입한다.
    실패 시 warning 만 출력하고 메쉬는 그대로 반환한다.

Examples:
    직접 사용::

        from pathlib import Path
        from naviertwin.core.cfd_reader.su2_reader import SU2Reader

        reader = SU2Reader()
        dataset = reader.read(Path("inv_NACA0012.su2"))

    팩토리 경유::

        from naviertwin.core.cfd_reader import ReaderFactory
        dataset = ReaderFactory.create_and_read(Path("inv_NACA0012.su2"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from naviertwin.core.cfd_reader._mesh_utils import meshio_to_cfd_dataset
from naviertwin.core.cfd_reader.base import BaseReader, CFDDataset
from naviertwin.core.cfd_reader.reader_factory import ReaderFactory
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


# SU2 셀 타입 코드 → VTK 셀 타입 코드 매핑
# https://su2code.github.io/docs_v7/Mesh-File/
_SU2_TO_VTK = {
    3: (2, 3),    # Line
    5: (3, 5),    # Triangle
    9: (4, 9),    # Quad
    10: (4, 10),  # Tetra
    12: (8, 12),  # Hexahedron
    13: (6, 13),  # Wedge (prism)
    14: (5, 14),  # Pyramid
}


@ReaderFactory.register
class SU2Reader(BaseReader):
    """SU2 네이티브 ASCII 메쉬 리더.

    SU2 ``.su2`` 파일과 sibling ``.csv`` 결과 파일을 읽는다.

    Attributes:
        supported_extensions: ``.su2`` 확장자를 지원한다.
    """

    supported_extensions: frozenset[str] = frozenset({".su2"})

    def read(self, path: Path) -> CFDDataset:
        """SU2 .su2 파일을 읽어 CFDDataset 을 반환한다.

        Args:
            path: .su2 파일 경로.

        Returns:
            파싱된 CFDDataset.

        Raises:
            FileNotFoundError: 경로가 존재하지 않는 경우.
            ValueError: 모든 파서가 실패한 경우.
        """
        if not path.exists():
            raise FileNotFoundError(f"파일이 존재하지 않습니다: {path}")

        logger.info("SU2 .su2 읽기 시작: %s", path)

        meshio_err: Exception | None = None
        ascii_err: Exception | None = None

        # 1. meshio
        try:
            dataset = self._read_with_meshio(path)
            self._attach_csv_fields(dataset, path)
            return dataset
        except ImportError:
            logger.info(
                "meshio 미설치 → ASCII 파서 폴백. 설치: pip install meshio"
            )
        except Exception as e:
            meshio_err = e
            logger.debug("meshio 폴백 실패: %s", e)

        # 2. SU2ASCIIParser
        try:
            dataset = SU2ASCIIParser(path).parse()
            self._attach_csv_fields(dataset, path)
            return dataset
        except Exception as e:
            ascii_err = e
            logger.debug("SU2ASCIIParser 실패: %s", e)

        raise ValueError(
            f"[SU2Reader] 모든 파서 실패: {path}\n"
            f"  1. meshio: {meshio_err}\n"
            f"  2. SU2ASCIIParser: {ascii_err}"
        )

    # ------------------------------------------------------------------
    # 내부 파서
    # ------------------------------------------------------------------

    def _read_with_meshio(self, path: Path) -> CFDDataset:
        try:
            import meshio
        except ImportError as exc:
            raise ImportError("meshio 미설치") from exc

        logger.debug("meshio 로 SU2 .su2 읽기: %s", path)
        mesh = meshio.read(str(path), file_format="su2")
        return meshio_to_cfd_dataset(mesh, str(path), "meshio/SU2")

    def _attach_csv_fields(self, dataset: CFDDataset, su2_path: Path) -> None:
        """sibling .csv 결과 파일이 있으면 point_data 에 병합한다.

        탐색 순서:
            1. <base>.csv
            2. <base>_restart.csv
            3. <base>_surface.csv

        실패 시 warning 후 반환.
        """
        candidates = [
            su2_path.with_suffix(".csv"),
            su2_path.with_name(f"{su2_path.stem}_restart.csv"),
            su2_path.with_name(f"{su2_path.stem}_surface.csv"),
            su2_path.with_name("restart_flow.csv"),
        ]
        csv_path = None
        candidate_idx = 0
        while candidate_idx < len(candidates):
            candidate = candidates[candidate_idx]
            if candidate.exists():
                csv_path = candidate
                break
            candidate_idx += 1
        if csv_path is None:
            logger.debug("sibling .csv 없음 — 메쉬만 로드됨: %s", su2_path)
            return

        logger.info("sibling .csv 발견: %s", csv_path)
        try:
            self._merge_csv_fields(dataset, csv_path)
        except Exception as e:
            logger.warning(".csv 필드 로드 실패 (무시): %s", e)

    def _merge_csv_fields(self, dataset: CFDDataset, csv_path: Path) -> None:
        """SU2 출력 .csv 를 읽어 point_data 에 추가한다.

        SU2 출력 .csv 헤더 예시::

            "PointID","x","y","z","Density","Momentum_x",...

        좌표 열은 스킵하고 나머지를 필드로 주입한다.
        """
        try:
            import numpy as np
        except ImportError:  # numpy 는 core 이므로 사실상 발생하지 않음
            return

        with csv_path.open("r", encoding="utf-8", errors="replace") as f:
            header_line = f.readline().strip()
        if not header_line:
            logger.warning(".csv 헤더가 비어 있습니다: %s", csv_path)
            return

        headers = []
        raw_headers = header_line.split(",")
        header_idx = 0
        while header_idx < len(raw_headers):
            headers.append(raw_headers[header_idx].strip().strip('"'))
            header_idx += 1
        data = np.loadtxt(
            csv_path, delimiter=",", skiprows=1, dtype=float, ndmin=2
        )
        if data.size == 0:
            logger.warning(".csv 데이터가 비어 있습니다: %s", csv_path)
            return

        n_points = dataset.n_points
        if data.shape[0] != n_points:
            logger.warning(
                ".csv 행 수(%d)가 mesh n_points(%d)와 다릅니다 — 필드 병합 생략",
                data.shape[0],
                n_points,
            )
            return

        mesh = dataset.mesh
        skip = {"pointid", "point_id", "x", "y", "z", "id"}
        merged_count = 0
        col_idx = 0
        while col_idx < len(headers):
            name = headers[col_idx]
            if name.lower() in skip:
                col_idx += 1
                continue
            if not hasattr(mesh, "point_data"):
                break
            mesh.point_data[name] = data[:, col_idx]
            merged_count += 1
            if name not in dataset.field_names:
                dataset.field_names.append(name)
            col_idx += 1

        dataset.metadata["csv_sidecar"] = str(csv_path)
        logger.debug(
            ".csv 필드 병합 완료: %s → %d fields",
            csv_path.name,
            merged_count,
        )


# ---------------------------------------------------------------------------
# SU2ASCIIParser — NDIME / NPOIN / NELEM / NMARK 섹션 직접 파싱
# ---------------------------------------------------------------------------


class SU2ASCIIParser:
    """SU2 네이티브 ASCII 메쉬 파서.

    섹션:
        - NDIME= n        : 공간 차원 (2 또는 3)
        - NELEM= n        : 체적 셀 수 + 연결성 라인
        - NPOIN= n        : 노드 수 + 좌표 라인
        - NMARK= n        : 경계 마커 수 (이후 MARKER_TAG / MARKER_ELEMS)

    Examples:
        >>> parser = SU2ASCIIParser(Path("mesh.su2"))
        >>> dataset = parser.parse()
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def parse(self) -> CFDDataset:
        """SU2 ASCII 파일을 파싱하여 CFDDataset 을 반환한다.

        Returns:
            파싱된 CFDDataset.

        Raises:
            ValueError: 필수 섹션이 없거나 파싱 실패 시.
        """
        try:
            import numpy as np
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(
                "SU2ASCIIParser 에 numpy, pyvista 가 필요합니다."
            ) from exc

        with self._path.open("r", encoding="utf-8", errors="replace") as f:
            raw_lines = f.readlines()

        lines = []
        raw_idx = 0
        while raw_idx < len(raw_lines):
            stripped = raw_lines[raw_idx].strip()
            if stripped and not stripped.lstrip().startswith("%"):
                lines.append(stripped)
            raw_idx += 1

        ndime = self._parse_keyword(lines, "NDIME")
        if ndime not in (2, 3):
            raise ValueError(
                f"SU2ASCIIParser: NDIME 섹션이 올바르지 않습니다 (값={ndime})"
            )

        # NPOIN 섹션
        idx = -1
        line_idx = 0
        while line_idx < len(lines):
            if lines[line_idx].upper().startswith("NPOIN"):
                idx = line_idx
                break
            line_idx += 1
        if idx < 0:
            raise ValueError("SU2ASCIIParser: NPOIN 섹션을 찾을 수 없습니다")

        n_poin = int(lines[idx].split("=")[1].split()[0])
        points = np.zeros((n_poin, 3), dtype=np.float64)
        k = 0
        while k < n_poin:
            parts = lines[idx + 1 + k].split()
            if ndime == 2:
                points[k, 0] = float(parts[0])
                points[k, 1] = float(parts[1])
                # z = 0
            else:
                points[k, 0] = float(parts[0])
                points[k, 1] = float(parts[1])
                points[k, 2] = float(parts[2])
            k += 1

        # NELEM 섹션 (옵션 — 없으면 포인트만 반환)
        cells: list[int] = []
        cell_types: list[int] = []
        elem_idx = -1
        line_idx = 0
        while line_idx < len(lines):
            if lines[line_idx].upper().startswith("NELEM"):
                elem_idx = line_idx
                break
            line_idx += 1
        if elem_idx >= 0:
            n_elem = int(lines[elem_idx].split("=")[1].split()[0])
            k = 0
            while k < n_elem:
                parts = lines[elem_idx + 1 + k].split()
                su2_type = int(parts[0])
                mapping = _SU2_TO_VTK.get(su2_type)
                if mapping is None:
                    k += 1
                    continue
                n_nodes, vtk_type = mapping
                node_ids = []
                node_idx = 0
                while node_idx < n_nodes:
                    node_ids.append(int(parts[1 + node_idx]))
                    node_idx += 1
                cells.extend([n_nodes, *node_ids])
                cell_types.append(vtk_type)
                k += 1

        metadata: dict[str, Any] = {
            "reader": "SU2ASCIIParser",
            "source_file": str(self._path),
            "ndime": ndime,
            "n_elem_parsed": len(cell_types),
        }

        if cells:
            mesh = pv.UnstructuredGrid(
                np.array(cells, dtype=np.int64),
                np.array(cell_types, dtype=np.uint8),
                points,
            )
        else:
            mesh = pv.PolyData(points).cast_to_unstructured_grid()

        logger.debug(
            "SU2ASCIIParser 완료: %d 노드, %d 셀", n_poin, len(cell_types)
        )

        return CFDDataset(
            mesh=mesh,
            time_steps=[0.0],
            field_names=[],
            metadata=metadata,
        )

    @staticmethod
    def _parse_keyword(lines: list[str], keyword: str) -> int:
        """`KEYWORD= n` 형식 라인을 찾아 정수값을 반환한다. 없으면 -1."""
        line_idx = 0
        while line_idx < len(lines):
            ln = lines[line_idx]
            if ln.upper().startswith(keyword.upper()):
                try:
                    return int(ln.split("=")[1].split()[0])
                except (IndexError, ValueError):
                    return -1
            line_idx += 1
        return -1
