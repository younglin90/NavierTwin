"""NavierTwin 전용 HDF5 파일 포맷 (.ntwin) 저장/로드 모듈.

`.ntwin` 파일은 HDF5 기반이며 ParaView 호환 VTKHDF 구조를 따른다.
타임스텝을 스트리밍 방식으로 append 할 수 있어 대용량 CFD 데이터에 적합하다.

파일 구조::

    root/
    ├── VTKHDF/                    # ParaView 호환 표준 그룹
    │   ├── attrs: Version=(2,0), Type="UnstructuredGrid"
    │   ├── Points                 shape (N_total, 3), dtype float32
    │   ├── Connectivity           flat cell connectivity array
    │   ├── Offsets                cell offset array
    │   ├── Types                  VTK cell type codes, uint8
    │   ├── NumberOfPoints         per-timestep 점 개수 배열
    │   ├── NumberOfCells          per-timestep 셀 개수 배열
    │   └── PointData/
    │       └── {field_name}       shape (N_total, ...), resizable
    └── NavierTwin/                # 확장 메타데이터
        ├── version                str "0.2.0"
        ├── time_steps             float64 배열
        └── field_names            JSON 인코딩 문자열 배열

Examples:
    쓰기::

        from pathlib import Path
        from naviertwin.core.export.ntwin_format import save_dataset

        save_dataset(dataset, Path("result.ntwin"))

    읽기::

        from naviertwin.core.export.ntwin_format import load_dataset

        dataset = load_dataset(Path("result.ntwin"))

    컨텍스트 매니저::

        with NTwinWriter(Path("result.ntwin")) as writer:
            writer.write_dataset(dataset)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

_NTWIN_VERSION = "0.2.0"
_VTKHDF_VERSION = (2, 0)

# h5py 가 없을 때 사용할 안내 메시지
_H5PY_MISSING_MSG = (
    "`.ntwin` 파일을 사용하려면 h5py 가 필요합니다.\n"
    "  pip install h5py"
)


def _require_h5py() -> Any:
    """h5py 를 import 하거나 친절한 에러를 발생시킨다.

    Returns:
        h5py 모듈.

    Raises:
        ImportError: h5py 가 설치되어 있지 않은 경우.
    """
    try:
        import h5py

        return h5py
    except ImportError as exc:
        raise ImportError(_H5PY_MISSING_MSG) from exc


def _require_numpy() -> Any:
    """numpy 를 import 한다.

    Returns:
        numpy 모듈.

    Raises:
        ImportError: numpy 가 설치되어 있지 않은 경우.
    """
    try:
        import numpy as np

        return np
    except ImportError as exc:
        raise ImportError("numpy 가 필요합니다: pip install numpy") from exc


def _require_pyvista() -> Any:
    """pyvista 를 import 한다.

    Returns:
        pyvista 모듈.

    Raises:
        ImportError: pyvista 가 설치되어 있지 않은 경우.
    """
    try:
        import pyvista as pv

        return pv
    except ImportError as exc:
        raise ImportError(
            "pyvista 가 필요합니다: pip install pyvista"
        ) from exc


# ---------------------------------------------------------------------------
# NTwinWriter
# ---------------------------------------------------------------------------


class NTwinWriter:
    """NavierTwin 프로젝트 파일 (.ntwin) 저장 클래스.

    HDF5 파일을 열어 VTKHDF 구조로 CFDDataset 또는 개별 스냅샷을 기록한다.
    컨텍스트 매니저로 사용하면 자동으로 파일이 닫힌다.

    Args:
        path: 저장할 .ntwin 파일 경로.

    Raises:
        ImportError: h5py 가 설치되어 있지 않은 경우.
    """

    def __init__(self, path: Path) -> None:
        h5py = _require_h5py()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(str(self._path), "w")
        self._initialized = False
        logger.debug("NTwinWriter 생성: %s", self._path)

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def write_dataset(self, dataset: Any) -> None:
        """CFDDataset 전체를 .ntwin 파일에 저장한다.

        단일 타임스텝 메쉬를 기준으로 VTKHDF 구조를 초기화하고
        모든 필드 데이터를 기록한다.

        Args:
            dataset: 저장할 :class:`~naviertwin.core.cfd_reader.base.CFDDataset`.
        """
        np = _require_numpy()
        mesh = dataset.mesh

        self._init_vtkhdf(mesh)
        self._write_topology(mesh)

        # 필드 데이터 저장
        point_data_grp = self._h5["VTKHDF/PointData"]
        for name in dataset.field_names:
            arr: Any = None
            if hasattr(mesh, "point_data") and name in mesh.point_data:
                arr = np.asarray(mesh.point_data[name], dtype=np.float32)
            elif hasattr(mesh, "cell_data") and name in mesh.cell_data:
                arr = np.asarray(mesh.cell_data[name], dtype=np.float32)
            if arr is not None:
                point_data_grp.create_dataset(name, data=arr)

        # NumberOfPoints/Cells
        n_pts = int(mesh.n_points) if hasattr(mesh, "n_points") else 0
        n_cls = int(mesh.n_cells) if hasattr(mesh, "n_cells") else 0
        vtk_grp = self._h5["VTKHDF"]
        vtk_grp.create_dataset(
            "NumberOfPoints",
            data=np.array([n_pts], dtype=np.int64),
        )
        vtk_grp.create_dataset(
            "NumberOfCells",
            data=np.array([n_cls], dtype=np.int64),
        )

        # NavierTwin 메타데이터
        self._write_naviertwin_meta(dataset.time_steps, dataset.field_names)
        logger.info(
            "write_dataset 완료: %d 필드, %d 타임스텝",
            len(dataset.field_names),
            len(dataset.time_steps),
        )

    def append_snapshot(self, mesh: Any, time_value: float) -> None:
        """타임스텝 하나를 기존 파일에 append 한다.

        파일이 초기화되지 않은 경우 토폴로지를 먼저 기록한다.

        Args:
            mesh: 추가할 ``pv.UnstructuredGrid`` 스냅샷.
            time_value: 이 스냅샷의 시간 값 [s].
        """
        np = _require_numpy()

        if not self._initialized:
            self._init_vtkhdf(mesh)
            self._write_topology(mesh)
            self._initialized = True

        vtk_grp = self._h5["VTKHDF"]
        point_data_grp = self._h5["VTKHDF/PointData"]

        n_pts = int(mesh.n_points) if hasattr(mesh, "n_points") else 0
        n_cls = int(mesh.n_cells) if hasattr(mesh, "n_cells") else 0

        # NumberOfPoints/Cells append
        for key, value in (
            ("NumberOfPoints", n_pts),
            ("NumberOfCells", n_cls),
        ):
            if key in vtk_grp:
                ds = vtk_grp[key]
                old_len = ds.shape[0]
                ds.resize(old_len + 1, axis=0)
                ds[old_len] = value
            else:
                vtk_grp.create_dataset(
                    key,
                    data=np.array([value], dtype=np.int64),
                    maxshape=(None,),
                )

        # 각 필드 append
        if hasattr(mesh, "point_data"):
            for name, arr_raw in mesh.point_data.items():
                arr = np.asarray(arr_raw, dtype=np.float32)
                if name in point_data_grp:
                    ds = point_data_grp[name]
                    old_len = ds.shape[0]
                    ds.resize(old_len + arr.shape[0], axis=0)
                    ds[old_len:] = arr
                else:
                    maxshape = (None,) + arr.shape[1:]
                    point_data_grp.create_dataset(
                        name, data=arr, maxshape=maxshape
                    )

        # NavierTwin 메타데이터 업데이트
        nt_grp = self._h5.require_group("NavierTwin")
        if "time_steps" in nt_grp:
            ts_ds = nt_grp["time_steps"]
            old_len = ts_ds.shape[0]
            ts_ds.resize(old_len + 1, axis=0)
            ts_ds[old_len] = time_value
        else:
            nt_grp.create_dataset(
                "time_steps",
                data=np.array([time_value], dtype=np.float64),
                maxshape=(None,),
            )

        logger.debug("append_snapshot: t=%.6f, n_pts=%d", time_value, n_pts)

    def close(self) -> None:
        """HDF5 파일을 닫는다."""
        if self._h5.id.valid:
            self._h5.close()
            logger.debug("NTwinWriter 닫힘: %s", self._path)

    def __enter__(self) -> "NTwinWriter":
        """컨텍스트 매니저 진입."""
        return self

    def __exit__(self, *args: Any) -> None:
        """컨텍스트 매니저 종료 시 파일을 닫는다."""
        self.close()

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _init_vtkhdf(self, mesh: Any) -> None:
        """VTKHDF 루트 그룹과 속성을 초기화한다.

        Args:
            mesh: 구조 참조용 메쉬.
        """
        if "VTKHDF" in self._h5:
            return
        vtk_grp = self._h5.require_group("VTKHDF")
        vtk_grp.attrs["Version"] = list(_VTKHDF_VERSION)
        vtk_grp.attrs["Type"] = "UnstructuredGrid"
        vtk_grp.require_group("PointData")
        self._initialized = True

    def _write_topology(self, mesh: Any) -> None:
        """메쉬 위상 정보(Points, Connectivity, Offsets, Types)를 기록한다.

        Args:
            mesh: ``pv.UnstructuredGrid`` 메쉬.
        """
        np = _require_numpy()
        vtk_grp = self._h5["VTKHDF"]

        # Points
        if hasattr(mesh, "points") and mesh.points is not None:
            pts = np.asarray(mesh.points, dtype=np.float32)
        else:
            pts = np.empty((0, 3), dtype=np.float32)
        if "Points" not in vtk_grp:
            vtk_grp.create_dataset("Points", data=pts)

        # Connectivity, Offsets, Types — pyvista UnstructuredGrid
        try:
            import pyvista as pv

            if isinstance(mesh, pv.UnstructuredGrid) and mesh.n_cells > 0:
                cells_arr = np.asarray(
                    mesh.cells, dtype=np.int64
                )
                cell_types = np.asarray(
                    mesh.celltypes, dtype=np.uint8
                )

                # connectivity (flat, VTK 형식: [npts, id0, id1, ...])
                if "Connectivity" not in vtk_grp:
                    vtk_grp.create_dataset(
                        "Connectivity", data=cells_arr
                    )

                # offsets (각 셀의 cells 배열 내 시작 위치)
                offsets = _compute_offsets(mesh)
                if "Offsets" not in vtk_grp:
                    vtk_grp.create_dataset(
                        "Offsets",
                        data=np.asarray(offsets, dtype=np.int64),
                    )

                if "Types" not in vtk_grp:
                    vtk_grp.create_dataset("Types", data=cell_types)
        except ImportError:
            pass

    def _write_naviertwin_meta(
        self, time_steps: list[float], field_names: list[str]
    ) -> None:
        """NavierTwin 확장 메타데이터 그룹을 기록한다.

        Args:
            time_steps: 타임스텝 리스트.
            field_names: 필드 이름 리스트.
        """
        np = _require_numpy()
        nt_grp = self._h5.require_group("NavierTwin")
        # version
        if "version" not in nt_grp:
            nt_grp.create_dataset("version", data=_NTWIN_VERSION)
        # time_steps
        if "time_steps" not in nt_grp:
            nt_grp.create_dataset(
                "time_steps",
                data=np.array(time_steps, dtype=np.float64),
                maxshape=(None,),
            )
        # field_names (JSON 인코딩)
        if "field_names" not in nt_grp:
            nt_grp.create_dataset(
                "field_names", data=json.dumps(field_names)
            )


# ---------------------------------------------------------------------------
# NTwinReader
# ---------------------------------------------------------------------------


class NTwinReader:
    """NavierTwin 프로젝트 파일 (.ntwin) 로드 클래스.

    HDF5 파일을 읽기 전용으로 열어 전체 데이터셋 또는 특정 타임스텝만
    로드하는 인터페이스를 제공한다.

    Args:
        path: 읽을 .ntwin 파일 경로.

    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우.
        ImportError: h5py 가 설치되어 있지 않은 경우.
    """

    def __init__(self, path: Path) -> None:
        h5py = _require_h5py()
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(
                f"파일이 존재하지 않습니다: {self._path}"
            )
        self._h5 = h5py.File(str(self._path), "r")
        logger.debug("NTwinReader 생성: %s", self._path)

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def read(self) -> Any:
        """전체 데이터셋을 :class:`CFDDataset` 으로 로드한다.

        Returns:
            로드된 :class:`~naviertwin.core.cfd_reader.base.CFDDataset`.

        Raises:
            ImportError: pyvista 가 설치되어 있지 않은 경우.
        """
        from naviertwin.core.cfd_reader.base import CFDDataset

        ts = self.time_steps
        fn = self.field_names

        # 마지막 타임스텝 메쉬 (또는 전체를 대표하는 단일 메쉬)
        idx = max(0, len(ts) - 1)
        mesh = self.read_timestep(idx)

        logger.info(
            "read 완료: %d 타임스텝, 필드=%s", len(ts), fn
        )
        return CFDDataset(
            mesh=mesh,
            time_steps=ts,
            field_names=fn,
            metadata={"source_file": str(self._path), "reader": "NTwinReader"},
        )

    def read_timestep(self, t_idx: int) -> Any:
        """특정 타임스텝 인덱스의 메쉬를 로드한다.

        Args:
            t_idx: 타임스텝 인덱스 (0-based).

        Returns:
            ``pv.UnstructuredGrid`` 메쉬.

        Raises:
            IndexError: 인덱스가 범위를 벗어난 경우.
            ImportError: pyvista 또는 numpy 가 없는 경우.
        """
        np = _require_numpy()
        pv = _require_pyvista()

        ts = self.time_steps
        if not (0 <= t_idx < len(ts)):
            raise IndexError(
                f"타임스텝 인덱스 {t_idx} 이(가) 범위를 벗어납니다 "
                f"(0 ~ {len(ts) - 1})."
            )

        vtk_grp = self._h5["VTKHDF"]

        # Points
        if "Points" in vtk_grp:
            pts = np.asarray(vtk_grp["Points"], dtype=np.float64)
        else:
            pts = np.empty((0, 3), dtype=np.float64)

        # Connectivity / Offsets / Types
        cells: Any = None
        cell_types: Any = None
        if "Connectivity" in vtk_grp and "Types" in vtk_grp:
            conn = np.asarray(vtk_grp["Connectivity"], dtype=np.int64)
            types = np.asarray(vtk_grp["Types"], dtype=np.uint8)
            cells = conn
            cell_types = types

        if cells is not None and cell_types is not None and len(pts) > 0:
            try:
                mesh = pv.UnstructuredGrid(cells, cell_types, pts)
            except Exception:
                mesh = pv.UnstructuredGrid()
        else:
            mesh = pv.UnstructuredGrid()

        # PointData 필드 로드
        if "PointData" in vtk_grp:
            pd_grp = vtk_grp["PointData"]
            for name in self.field_names:
                if name in pd_grp:
                    arr = np.asarray(pd_grp[name])
                    mesh.point_data[name] = arr

        logger.debug("read_timestep t_idx=%d 완료", t_idx)
        return mesh

    @property
    def time_steps(self) -> list[float]:
        """파일에 저장된 타임스텝 리스트.

        Returns:
            float 타임스텝 값 리스트.
        """
        np = _require_numpy()
        try:
            ts_raw = self._h5["NavierTwin/time_steps"]
            return list(np.asarray(ts_raw, dtype=np.float64))
        except KeyError:
            return [0.0]

    @property
    def field_names(self) -> list[str]:
        """파일에 저장된 필드 이름 리스트.

        Returns:
            필드 이름 문자열 리스트.
        """
        try:
            raw = self._h5["NavierTwin/field_names"]
            decoded = raw[()].decode("utf-8") if hasattr(raw[()], "decode") else str(raw[()])
            return json.loads(decoded)
        except (KeyError, json.JSONDecodeError, AttributeError):
            # PointData 그룹에서 직접 수집
            if "VTKHDF/PointData" in self._h5:
                return sorted(self._h5["VTKHDF/PointData"].keys())
            return []

    def close(self) -> None:
        """HDF5 파일을 닫는다."""
        if self._h5.id.valid:
            self._h5.close()
            logger.debug("NTwinReader 닫힘: %s", self._path)

    def __enter__(self) -> "NTwinReader":
        """컨텍스트 매니저 진입."""
        return self

    def __exit__(self, *args: Any) -> None:
        """컨텍스트 매니저 종료 시 파일을 닫는다."""
        self.close()


# ---------------------------------------------------------------------------
# 편의 함수
# ---------------------------------------------------------------------------


def save_dataset(dataset: Any, path: Path) -> None:
    """CFDDataset 을 .ntwin 파일로 저장하는 편의 함수.

    Args:
        dataset: 저장할 :class:`~naviertwin.core.cfd_reader.base.CFDDataset`.
        path: 저장 경로 (.ntwin 확장자 권장).

    Raises:
        ImportError: h5py 가 설치되어 있지 않은 경우.
    """
    with NTwinWriter(Path(path)) as writer:
        writer.write_dataset(dataset)
    logger.info("save_dataset 완료: %s", path)


def load_dataset(path: Path) -> Any:
    """`.ntwin` 파일을 CFDDataset 으로 로드하는 편의 함수.

    Args:
        path: 로드할 .ntwin 파일 경로.

    Returns:
        로드된 :class:`~naviertwin.core.cfd_reader.base.CFDDataset`.

    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우.
        ImportError: h5py 가 설치되어 있지 않은 경우.
    """
    with NTwinReader(Path(path)) as reader:
        dataset = reader.read()
    logger.info("load_dataset 완료: %s", path)
    return dataset


# ---------------------------------------------------------------------------
# 내부 유틸리티
# ---------------------------------------------------------------------------


def _compute_offsets(mesh: Any) -> list[int]:
    """pv.UnstructuredGrid 의 VTK cells 배열에서 offset 배열을 계산한다.

    VTK cells 배열 형식: [n0, id0_0, id0_1, ..., n1, id1_0, ...]
    offset[i] = i번째 셀의 cells 배열 내 시작 인덱스.

    Args:
        mesh: ``pv.UnstructuredGrid`` 인스턴스.

    Returns:
        정수 offset 리스트 (길이: n_cells + 1).
    """
    try:
        import numpy as np

        cells = np.asarray(mesh.cells)
        offsets: list[int] = [0]
        idx = 0
        while idx < len(cells):
            n = int(cells[idx])
            idx += n + 1
            offsets.append(idx)
        return offsets
    except Exception:
        return [0]
