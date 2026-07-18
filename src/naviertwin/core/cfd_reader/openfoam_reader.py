"""OpenFOAM 케이스 리더 모듈.

pyvista.POpenFOAMReader 를 우선 시도하고, 실패 시 ofpp 라이브러리로
폴백하여 OpenFOAM 케이스 디렉토리 또는 .foam 파일을 읽는다.

지원 경로 형식:
    - ``/case/cavity.foam`` : .foam 더미 파일
    - ``/case/cavity/``     : 케이스 디렉토리 (자동으로 .foam 생성)

Examples:
    직접 사용::

        from pathlib import Path
        from naviertwin.core.cfd_reader.openfoam_reader import OpenFOAMReader

        reader = OpenFOAMReader()
        dataset = reader.read(Path("/cases/cavity"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from naviertwin.core.cfd_reader.base import BaseReader, CFDDataset
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class OpenFOAMReader(BaseReader):
    """OpenFOAM 케이스 리더.

    우선순위:
    1. ``pyvista.POpenFOAMReader`` — .foam 더미 파일 필요, PyVista 설치 필요.
    2. ``ofpp`` — 바이너리/ASCII OpenFOAM 포맷 직접 파싱, PyVista 불필요.

    두 라이브러리 모두 없으면 :exc:`ImportError` 를 발생시킨다.

    Attributes:
        supported_extensions: ``.foam``, ``.openfoam`` 확장자를 지원한다.
    """

    supported_extensions: frozenset[str] = frozenset({".foam", ".openfoam"})

    # ------------------------------------------------------------------
    # BaseReader 인터페이스
    # ------------------------------------------------------------------

    def read(self, path: Path) -> CFDDataset:
        """OpenFOAM 케이스를 읽어 :class:`CFDDataset` 으로 반환한다.

        ``path`` 가 디렉토리이면 .foam 더미 파일을 자동 생성한다.

        Args:
            path: 케이스 디렉토리 또는 .foam 파일 경로.

        Returns:
            파싱된 :class:`CFDDataset`.

        Raises:
            FileNotFoundError: 경로가 존재하지 않는 경우.
            ImportError: pyvista 와 ofpp 모두 없는 경우.
        """
        if not path.exists():
            raise FileNotFoundError(f"경로가 존재하지 않습니다: {path}")

        foam_file = self._find_foam_file(path)
        case_dir = foam_file.parent

        logger.info("OpenFOAM 케이스 읽기 시작: %s", case_dir)

        # pyvista 우선 시도
        try:
            return self._read_with_pyvista(foam_file)
        except ImportError:
            logger.warning(
                "pyvista 를 불러올 수 없습니다. ofpp 로 폴백합니다."
            )

        # ofpp 폴백
        try:
            return self._read_with_ofpp(case_dir)
        except ImportError as exc:
            raise ImportError(
                "OpenFOAM 케이스를 읽으려면 pyvista 또는 ofpp 가 필요합니다.\n"
                "  pip install pyvista\n"
                "  pip install ofpp"
            ) from exc

    # ------------------------------------------------------------------
    # pyvista 기반 읽기
    # ------------------------------------------------------------------

    def _read_with_pyvista(self, foam_file: Path) -> CFDDataset:
        """pyvista.POpenFOAMReader 로 OpenFOAM 케이스를 읽는다.

        Args:
            foam_file: .foam 더미 파일 경로.

        Returns:
            파싱된 :class:`CFDDataset`.

        Raises:
            ImportError: pyvista 가 설치되어 있지 않은 경우.
        """
        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError("pyvista 가 설치되어 있지 않습니다.") from exc

        case_dir = foam_file.parent
        logger.debug("pyvista.POpenFOAMReader 로 읽기: %s", foam_file)

        reader = pv.POpenFOAMReader(str(foam_file))
        reader.enable_all_cell_arrays()
        reader.enable_all_point_arrays()
        try:
            # 경계 패치 블록도 함께 읽는다 (internalMesh 는 기본 활성).
            reader.enable_all_patch_arrays()
        except Exception:  # noqa: BLE001 — 패치 미지원 리더/파일 보호
            logger.debug("패치 배열 활성화 실패 — 경계 패치 없이 진행합니다.")

        time_steps = self._detect_time_steps(case_dir)
        field_names: list[str] = []
        mesh: Any = None
        multi_block: Any = None

        if time_steps:
            # 마지막 타임스텝으로 메쉬 로드
            reader.set_active_time_value(time_steps[-1])
            multi_block = reader.read()
            mesh = _extract_internal_mesh(multi_block)
            if mesh is not None:
                field_names = _collect_field_names(mesh)
        else:
            # 타임스텝 없으면 그냥 읽기
            multi_block = reader.read()
            mesh = _extract_internal_mesh(multi_block)
            if mesh is not None:
                field_names = _collect_field_names(mesh)
            time_steps = [0.0]

        if mesh is None:
            # 빈 UnstructuredGrid 반환
            mesh = pv.UnstructuredGrid()

        metadata: dict[str, Any] = {
            "reader": "pyvista.POpenFOAMReader",
            "foam_file": str(foam_file),
            "case_dir": str(case_dir),
        }

        try:
            patch_meshes = _extract_boundary_patches(multi_block)
        except Exception:  # noqa: BLE001 — 경계 정보 없는 파일 보호
            logger.debug("경계 패치 추출 실패 — 패치 메타데이터를 생략합니다.")
            patch_meshes = {}

        if patch_meshes:
            metadata["boundary_patches"] = {
                name: {
                    "n_cells": int(patch.n_cells),
                    "n_points": int(patch.n_points),
                }
                for name, patch in patch_meshes.items()
            }
            metadata["boundary_patch_meshes"] = patch_meshes
            logger.info(
                "경계 패치 %d개 보존: %s",
                len(patch_meshes),
                sorted(patch_meshes),
            )

        return CFDDataset(
            mesh=mesh,
            time_steps=time_steps,
            field_names=field_names,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # ofpp 기반 읽기
    # ------------------------------------------------------------------

    def _read_with_ofpp(self, case_dir: Path) -> CFDDataset:
        """ofpp 라이브러리로 OpenFOAM 케이스를 읽는다.

        Args:
            case_dir: OpenFOAM 케이스 루트 디렉토리.

        Returns:
            파싱된 :class:`CFDDataset`.

        Raises:
            ImportError: ofpp 가 설치되어 있지 않은 경우.
        """
        try:
            import ofpp  # type: ignore[import-untyped]  # noqa: F401  (availability probe)
        except ImportError as exc:
            raise ImportError("ofpp 가 설치되어 있지 않습니다.") from exc

        try:
            import numpy as np  # noqa: F401  (availability probe)
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(
                "ofpp 폴백에는 numpy 와 pyvista 가 필요합니다."
            ) from exc

        logger.debug("ofpp 로 읽기: %s", case_dir)

        time_steps = self._detect_time_steps(case_dir)
        if not time_steps:
            time_steps = [0.0]

        # 마지막 타임스텝 선택
        last_time = time_steps[-1]
        time_dir = str(last_time).rstrip("0").rstrip(".")
        if "." not in time_dir:
            time_dir = str(int(last_time))

        field_names = self._detect_field_names(case_dir, time_dir)

        # 간소화 메쉬 구성: numpy 배열만으로 UnstructuredGrid 생성
        mesh = pv.UnstructuredGrid()

        return CFDDataset(
            mesh=mesh,
            time_steps=time_steps,
            field_names=field_names,
            metadata={
                "reader": "ofpp",
                "case_dir": str(case_dir),
                "last_time_dir": time_dir,
            },
        )

    # ------------------------------------------------------------------
    # 헬퍼 메서드
    # ------------------------------------------------------------------

    def _find_foam_file(self, path: Path) -> Path:
        """경로에서 .foam 파일을 찾거나 생성한다.

        ``path`` 가 .foam 파일이면 그대로 반환한다.
        디렉토리이면 기존 .foam 파일을 검색하고, 없으면 생성한다.

        Args:
            path: .foam 파일 또는 케이스 디렉토리 경로.

        Returns:
            .foam 파일 경로.
        """
        if path.is_file():
            return path

        # 디렉토리에서 기존 .foam 파일 검색
        foam_file = next(path.glob("*.foam"), None)
        if foam_file is not None:
            logger.debug("기존 .foam 파일 발견: %s", foam_file)
            return foam_file

        foam_file = next(path.glob("*.OpenFOAM"), None)
        if foam_file is not None:
            logger.debug("기존 .OpenFOAM 파일 발견: %s", foam_file)
            return foam_file

        return self._create_foam_file(path)

    def _create_foam_file(self, case_dir: Path) -> Path:
        """케이스 디렉토리에 빈 .foam 더미 파일을 생성한다.

        OpenFOAM 케이스 이름은 디렉토리 이름을 사용한다.

        Args:
            case_dir: OpenFOAM 케이스 루트 디렉토리.

        Returns:
            생성된 .foam 파일 경로.
        """
        foam_file = case_dir / f"{case_dir.name}.foam"
        foam_file.touch(exist_ok=True)
        logger.debug(".foam 더미 파일 생성: %s", foam_file)
        return foam_file

    def _detect_time_steps(self, case_dir: Path) -> list[float]:
        """케이스 디렉토리에서 타임스텝 목록을 탐지한다.

        숫자 이름을 가진 하위 디렉토리를 타임스텝으로 인식한다.
        ``constant``, ``system``, ``processor*`` 디렉토리는 제외한다.

        Args:
            case_dir: OpenFOAM 케이스 루트 디렉토리.

        Returns:
            오름차순 정렬된 타임스텝 float 리스트.
        """
        _SKIP: frozenset[str] = frozenset({"constant", "system", "0.orig"})
        time_steps: list[float] = []

        children = list(case_dir.iterdir())
        child_idx = 0
        while child_idx < len(children):
            child = children[child_idx]
            if not child.is_dir():
                child_idx += 1
                continue
            name = child.name
            if name in _SKIP or name.startswith("processor"):
                child_idx += 1
                continue
            try:
                t = float(name)
                time_steps.append(t)
            except ValueError:
                pass
            child_idx += 1

        time_steps.sort()
        logger.debug("탐지된 타임스텝: %s", time_steps)
        return time_steps

    def _detect_field_names(
        self, case_dir: Path, time_dir: str
    ) -> list[str]:
        """특정 타임스텝 디렉토리에서 필드 이름 목록을 반환한다.

        Args:
            case_dir: OpenFOAM 케이스 루트 디렉토리.
            time_dir: 타임스텝 디렉토리 이름 (예: "1", "0.5").

        Returns:
            필드 이름 문자열 리스트.
        """
        t_path = case_dir / time_dir
        if not t_path.is_dir():
            logger.warning("타임 디렉토리 없음: %s", t_path)
            return []

        field_names = []
        paths = list(t_path.iterdir())
        path_idx = 0
        while path_idx < len(paths):
            p = paths[path_idx]
            if p.is_file() and not p.name.startswith("."):
                field_names.append(p.name)
            path_idx += 1
        logger.debug("필드 이름(%s): %s", time_dir, field_names)
        return sorted(field_names)


# ---------------------------------------------------------------------------
# 모듈 수준 헬퍼 함수 (pyvista MultiBlock 처리)
# ---------------------------------------------------------------------------


def _extract_internal_mesh(multi_block: Any) -> Any:
    """MultiBlock 에서 internalMesh 블록을 우선 추출한다.

    패치 배열이 활성화된 상태에서는 ``combine()`` 이 경계 표면까지 병합해
    버리므로, ``internalMesh`` 블록이 있으면 그것만 UnstructuredGrid 로
    반환한다. 없으면 기존 :func:`_extract_unstructured_grid` 동작으로
    폴백한다.

    Args:
        multi_block: pyvista MultiBlock 또는 DataSet 객체.

    Returns:
        UnstructuredGrid 또는 None.
    """
    try:
        import pyvista as pv
    except ImportError:
        return None

    if isinstance(multi_block, pv.MultiBlock):
        try:
            keys = list(multi_block.keys())
        except Exception:  # noqa: BLE001 — 이름 없는 블록 보호
            keys = []
        if "internalMesh" in keys:
            internal = multi_block["internalMesh"]
            if internal is not None:
                if isinstance(internal, pv.UnstructuredGrid):
                    return internal
                try:
                    return internal.cast_to_unstructured_grid()
                except Exception:  # noqa: BLE001 — 캐스팅 불가 시 폴백
                    logger.debug("internalMesh 캐스팅 실패 — combine 으로 폴백")

    return _extract_unstructured_grid(multi_block)


def _extract_boundary_patches(multi_block: Any) -> dict[str, Any]:
    """MultiBlock 의 boundary 블록에서 패치별 표면 메쉬를 추출한다.

    pyvista.POpenFOAMReader 가 패치 배열을 활성화한 채 읽으면 결과
    MultiBlock 에 ``boundary`` 라는 중첩 MultiBlock 이 생기고, 그 안에
    패치 이름별 표면(PolyData)이 담긴다.

    Args:
        multi_block: pyvista MultiBlock 또는 DataSet 객체.

    Returns:
        {패치 이름: PolyData 표면} 딕셔너리. 경계 정보가 없으면 빈 dict.
    """
    try:
        import pyvista as pv
    except ImportError:
        return {}

    if not isinstance(multi_block, pv.MultiBlock):
        return {}

    try:
        keys = list(multi_block.keys())
    except Exception:  # noqa: BLE001 — 이름 없는 블록 보호
        return {}
    if "boundary" not in keys:
        return {}

    boundary = multi_block["boundary"]
    if not isinstance(boundary, pv.MultiBlock):
        return {}

    patches: dict[str, Any] = {}
    for name in boundary.keys():
        if name is None:
            continue
        block = boundary[name]
        if block is None or int(getattr(block, "n_points", 0)) == 0:
            continue
        if not isinstance(block, pv.PolyData):
            try:
                block = block.extract_surface()
            except Exception:  # noqa: BLE001 — 표면 추출 불가 패치는 생략
                logger.debug("패치 '%s' 표면 추출 실패 — 생략", name)
                continue
        patches[str(name)] = block
    return patches


def _extract_unstructured_grid(multi_block: Any) -> Any:
    """MultiBlock 에서 첫 번째 UnstructuredGrid 를 추출한다.

    Args:
        multi_block: pyvista MultiBlock 또는 DataSet 객체.

    Returns:
        UnstructuredGrid 또는 None.
    """
    try:
        import pyvista as pv
    except ImportError:
        return None

    if isinstance(multi_block, pv.UnstructuredGrid):
        return multi_block

    if hasattr(multi_block, "combine"):
        combined = multi_block.combine()
        if isinstance(combined, pv.UnstructuredGrid):
            return combined
        return combined.cast_to_unstructured_grid()

    return None


def _collect_field_names(mesh: Any) -> list[str]:
    """메쉬의 point_data 와 cell_data 에서 필드 이름을 수집한다.

    Args:
        mesh: pyvista DataSet 객체.

    Returns:
        중복 없는 정렬된 필드 이름 리스트.
    """
    names: set[str] = set()
    if hasattr(mesh, "point_data"):
        names.update(mesh.point_data.keys())
    if hasattr(mesh, "cell_data"):
        names.update(mesh.cell_data.keys())
    return sorted(names)
