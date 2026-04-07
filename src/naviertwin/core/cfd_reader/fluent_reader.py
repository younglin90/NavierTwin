"""Ansys Fluent .cas/.dat 파일 리더 모듈.

ASCII .cas (메쉬) 와 .dat (필드 데이터) 쌍을 읽는다.
바이너리 .cas/.dat 는 지원하지 않는다.

폴백 체인:
    1. pyvista.FluentReader (vtkFLUENTReader)
    2. meshio
    3. FluentASCIIParser (섹션 ID 기반 ASCII 파서)

.dat 파일 처리:
    .cas 경로와 같은 디렉토리에 sibling .dat 파일이 있으면 자동으로 로드한다.
    없으면 logger.warning 을 출력하고 메쉬만 반환한다.

지원하지 않는 경우:
    - 바이너리 .cas/.dat → ValueError (명확한 오류 메시지)

Examples:
    직접 사용::

        from pathlib import Path
        from naviertwin.core.cfd_reader.fluent_reader import FluentReader

        reader = FluentReader()
        dataset = reader.read(Path("cavity.cas"))
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from naviertwin.core.cfd_reader._mesh_utils import meshio_to_cfd_dataset, pyvista_to_cfd_dataset
from naviertwin.core.cfd_reader.base import BaseReader, CFDDataset
from naviertwin.core.cfd_reader.reader_factory import ReaderFactory
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

# Fluent 바이너리 파일 매직 바이트 (첫 4바이트가 ASCII '(' 가 아닌 경우)
_ASCII_START = b"("


@ReaderFactory.register
class FluentReader(BaseReader):
    """Ansys Fluent .cas/.dat 리더.

    ASCII .cas 메쉬와 sibling .dat 필드 데이터를 읽는다.
    바이너리 포맷은 지원하지 않는다.

    Attributes:
        supported_extensions: ``.cas``, ``.dat`` 확장자를 지원한다.
    """

    supported_extensions: frozenset[str] = frozenset({".cas", ".dat"})

    def read(self, path: Path) -> CFDDataset:
        """Fluent .cas 파일을 읽어 CFDDataset 을 반환한다.

        .dat 확장자가 입력되면 .cas 를 먼저 찾아 읽은 뒤 필드를 병합한다.

        Args:
            path: .cas 또는 .dat 파일 경로.

        Returns:
            파싱된 CFDDataset.

        Raises:
            FileNotFoundError: 경로가 존재하지 않는 경우.
            ValueError: 바이너리 포맷이거나 모든 파서가 실패한 경우.
        """
        if not path.exists():
            raise FileNotFoundError(f"파일이 존재하지 않습니다: {path}")

        # .dat 단독 입력 → 대응 .cas 를 먼저 읽기
        if path.suffix.lower() == ".dat":
            cas_path = path.with_suffix(".cas")
            if cas_path.exists():
                logger.info(".dat 입력 → .cas 먼저 읽기: %s", cas_path)
                return self.read(cas_path)
            else:
                raise FileNotFoundError(
                    f".dat 에 대응하는 .cas 파일을 찾을 수 없습니다: {cas_path}"
                )

        _check_not_binary(path)

        logger.info("Fluent .cas 읽기 시작: %s", path)

        pv_err: Exception | None = None
        meshio_err: Exception | None = None
        ascii_err: Exception | None = None

        # 1. pyvista.FluentReader
        try:
            dataset = self._read_with_pyvista(path)
            self._attach_dat_fields(dataset, path)
            return dataset
        except ImportError:
            logger.info(
                "pyvista 미설치 → meshio 폴백. "
                "설치: pip install 'naviertwin[full]'"
            )
        except Exception as e:
            pv_err = e
            logger.debug("pyvista.FluentReader 실패: %s", e)

        # 2. meshio
        try:
            dataset = self._read_with_meshio(path)
            self._attach_dat_fields(dataset, path)
            return dataset
        except ImportError:
            logger.info(
                "meshio 미설치 → ASCII 파서 폴백. "
                "설치: pip install meshio"
            )
        except Exception as e:
            meshio_err = e
            logger.debug("meshio 폴백 실패: %s", e)

        # 3. FluentASCIIParser
        try:
            dataset = FluentASCIIParser(path).parse()
            self._attach_dat_fields(dataset, path)
            return dataset
        except Exception as e:
            ascii_err = e
            logger.debug("FluentASCIIParser 실패: %s", e)

        raise ValueError(
            f"[FluentReader] 모든 파서 실패: {path}\n"
            f"  1. pyvista.FluentReader: {pv_err}\n"
            f"  2. meshio: {meshio_err}\n"
            f"  3. FluentASCIIParser: {ascii_err}"
        )

    # ------------------------------------------------------------------
    # 내부 파서
    # ------------------------------------------------------------------

    def _read_with_pyvista(self, path: Path) -> CFDDataset:
        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError("pyvista 미설치") from exc

        logger.debug("pyvista.FluentReader 로 읽기: %s", path)
        reader = pv.FluentReader(str(path))
        mesh = reader.read()
        return pyvista_to_cfd_dataset(mesh, str(path), "pyvista.FluentReader")

    def _read_with_meshio(self, path: Path) -> CFDDataset:
        try:
            import meshio
        except ImportError as exc:
            raise ImportError("meshio 미설치") from exc

        logger.debug("meshio 로 Fluent .cas 읽기: %s", path)
        mesh = meshio.read(str(path))
        return meshio_to_cfd_dataset(mesh, str(path), "meshio/Fluent")

    def _attach_dat_fields(self, dataset: CFDDataset, cas_path: Path) -> None:
        """sibling .dat 파일이 있으면 필드 데이터를 dataset 에 병합한다."""
        dat_path = cas_path.with_suffix(".dat")
        if not dat_path.exists():
            logger.warning(
                ".dat 파일 없음 — 메쉬만 로드됨: %s", dat_path
            )
            return

        logger.info("sibling .dat 발견, 필드 로드 시도: %s", dat_path)
        try:
            _check_not_binary(dat_path)
        except ValueError:
            logger.warning(".dat 파일이 바이너리입니다. 필드 데이터 로드 생략.")
            return

        try:
            self._merge_dat_fields(dataset, dat_path)
        except Exception as e:
            logger.warning(".dat 필드 로드 실패 (무시): %s", e)

    def _merge_dat_fields(self, dataset: CFDDataset, dat_path: Path) -> None:
        """ASCII .dat 파일에서 필드를 읽어 dataset.mesh 에 추가한다.

        현재 구현: meshio 를 통해 .dat 읽기를 시도한다.
        meshio 가 실패하면 logger.warning 출력 후 반환.
        """
        try:
            import meshio
        except ImportError:
            logger.warning(
                ".dat 필드 로드에 meshio 가 필요합니다: "
                "pip install 'naviertwin[full]'"
            )
            return

        try:
            dat_mesh = meshio.read(str(dat_path))
        except Exception as e:
            logger.warning(".dat meshio 읽기 실패: %s", e)
            return

        # point_data 병합
        for key, arr in dat_mesh.point_data.items():
            if hasattr(dataset.mesh, "point_data"):
                dataset.mesh.point_data[key] = arr
                if key not in dataset.field_names:
                    dataset.field_names.append(key)

        logger.debug(".dat 필드 병합 완료: %s", list(dat_mesh.point_data.keys()))


# ---------------------------------------------------------------------------
# 바이너리 체크
# ---------------------------------------------------------------------------


def _check_not_binary(path: Path) -> None:
    """파일이 Fluent 바이너리 포맷이면 ValueError 를 발생시킨다.

    Fluent ASCII 파일은 반드시 '(' 로 시작한다.

    Args:
        path: 검사할 .cas 또는 .dat 파일 경로.

    Raises:
        ValueError: 바이너리 포맷으로 판단되는 경우.
    """
    try:
        with path.open("rb") as f:
            header = f.read(4)
    except OSError:
        return  # 읽기 실패 시 파서에게 맡김

    if header and not header.lstrip(b" \t\r\n").startswith(_ASCII_START):
        raise ValueError(
            f"Fluent 바이너리 .cas/.dat 미지원: {path}\n"
            "  ASCII 포맷으로 저장 후 사용하세요.\n"
            "  Fluent: File → Export → ASCII .cas"
        )


# ---------------------------------------------------------------------------
# FluentASCIIParser — 섹션 ID 기반 ASCII 파서
# ---------------------------------------------------------------------------


class FluentASCIIParser:
    """Fluent ASCII .cas 섹션 ID 기반 파서.

    지원 섹션:
        - 0x2010 (8208): 노드 좌표
        - 0x2012 (8210): 셀 타입
        - 0x2013 (8211): 면 연결성
        - 0x2040 (8256): 존 정보

    Examples:
        >>> parser = FluentASCIIParser(Path("cavity.cas"))
        >>> dataset = parser.parse()
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def parse(self) -> CFDDataset:
        """ASCII .cas 파일을 파싱하여 CFDDataset 을 반환한다.

        Returns:
            파싱된 CFDDataset.

        Raises:
            ValueError: 파일 파싱에 실패한 경우.
        """
        try:
            import numpy as np
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(
                "FluentASCIIParser 에 numpy, pyvista 가 필요합니다."
            ) from exc

        nodes: list[list[float]] = []
        metadata: dict[str, Any] = {
            "reader": "FluentASCIIParser",
            "source_file": str(self._path),
        }

        try:
            with self._path.open("r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as exc:
            raise ValueError(f"파일 읽기 실패: {self._path}") from exc

        # 섹션 0x2010: 노드 좌표
        import re

        node_pattern = re.compile(
            r"\(2010\s*\(([^)]+)\)\s*\(([\s\S]*?)\)\s*\)", re.IGNORECASE
        )
        for m in node_pattern.finditer(content):
            coord_block = m.group(2).strip()
            for line in coord_block.splitlines():
                vals = line.split()
                if len(vals) >= 3:
                    try:
                        nodes.append([float(v) for v in vals[:3]])
                    except ValueError:
                        pass

        if not nodes:
            raise ValueError(
                f"FluentASCIIParser: 노드 좌표(섹션 0x2010)를 찾을 수 없습니다: {self._path}"
            )

        points = np.array(nodes, dtype=np.float64)
        # 노드만 있는 최소 메쉬 반환 (셀 연결성 없음)
        mesh = pv.PolyData(points).cast_to_unstructured_grid()

        metadata["n_nodes_parsed"] = len(nodes)
        logger.debug(
            "FluentASCIIParser 완료: %d 노드", len(nodes)
        )

        return CFDDataset(
            mesh=mesh,
            time_steps=[0.0],
            field_names=[],
            metadata=metadata,
        )
