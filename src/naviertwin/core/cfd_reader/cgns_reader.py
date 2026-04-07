"""CGNS (.cgns) 파일 리더 모듈.

CGNS (CFD General Notation System) HDF5 파일을 읽는다.

폴백 체인:
    1. pyvista.CGNSReader (vtkCGNSReader)
    2. pyCGNS — CGNS.MAP.load()
    3. h5py — HDF5 직접 파싱
    4. meshio

Examples:
    직접 사용::

        from pathlib import Path
        from naviertwin.core.cfd_reader.cgns_reader import CGNSReader

        reader = CGNSReader()
        dataset = reader.read(Path("mesh.cgns"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from naviertwin.core.cfd_reader._mesh_utils import meshio_to_cfd_dataset, pyvista_to_cfd_dataset
from naviertwin.core.cfd_reader.base import BaseReader, CFDDataset
from naviertwin.core.cfd_reader.reader_factory import ReaderFactory
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


@ReaderFactory.register
class CGNSReader(BaseReader):
    """CGNS 파일 리더.

    폴백 체인: pyvista.CGNSReader → CGNS.MAP (pyCGNS) → h5py → meshio.

    Attributes:
        supported_extensions: ``.cgns`` 확장자를 지원한다.
    """

    supported_extensions: frozenset[str] = frozenset({".cgns"})

    def read(self, path: Path) -> CFDDataset:
        """CGNS 파일을 읽어 CFDDataset 을 반환한다.

        Args:
            path: .cgns 파일 경로.

        Returns:
            파싱된 CFDDataset.

        Raises:
            FileNotFoundError: 경로가 존재하지 않는 경우.
            ValueError: 모든 파서가 실패한 경우.
        """
        if not path.exists():
            raise FileNotFoundError(f"파일이 존재하지 않습니다: {path}")

        logger.info("CGNS 파일 읽기 시작: %s", path)

        pv_err: Exception | None = None
        cgns_err: Exception | None = None
        h5_err: Exception | None = None
        meshio_err: Exception | None = None

        # 1. pyvista.CGNSReader
        try:
            return self._read_with_pyvista(path)
        except ImportError:
            logger.info(
                "pyvista 미설치 → pyCGNS 폴백. "
                "설치: pip install 'naviertwin[full]'"
            )
        except Exception as e:
            pv_err = e
            logger.debug("pyvista.CGNSReader 실패: %s", e)

        # 2. pyCGNS (CGNS.MAP)
        try:
            return self._read_with_pycgns(path)
        except ImportError:
            logger.info(
                "pyCGNS 미설치 → h5py 폴백. "
                "설치: pip install 'naviertwin[full]'"
            )
        except Exception as e:
            cgns_err = e
            logger.debug("pyCGNS 폴백 실패: %s", e)

        # 3. h5py
        try:
            return self._read_with_h5py(path)
        except ImportError:
            logger.info(
                "h5py 미설치 → meshio 폴백. "
                "설치: pip install h5py"
            )
        except Exception as e:
            h5_err = e
            logger.debug("h5py 폴백 실패: %s", e)

        # 4. meshio
        try:
            return self._read_with_meshio(path)
        except ImportError:
            pass
        except Exception as e:
            meshio_err = e
            logger.debug("meshio 폴백 실패: %s", e)

        raise ValueError(
            f"[CGNSReader] 모든 파서 실패 (pyCGNS/h5py/meshio): {path}\n"
            f"  1. pyvista.CGNSReader: {pv_err}\n"
            f"  2. pyCGNS (CGNS.MAP): {cgns_err}\n"
            f"  3. h5py: {h5_err}\n"
            f"  4. meshio: {meshio_err}"
        )

    # ------------------------------------------------------------------
    # 내부 파서
    # ------------------------------------------------------------------

    def _read_with_pyvista(self, path: Path) -> CFDDataset:
        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError("pyvista 미설치") from exc

        logger.debug("pyvista.CGNSReader 로 읽기: %s", path)
        reader = pv.CGNSReader(str(path))
        reader.enable_all_bases()
        reader.enable_all_families()
        mesh = reader.read()
        return pyvista_to_cfd_dataset(mesh, str(path), "pyvista.CGNSReader")

    def _read_with_pycgns(self, path: Path) -> CFDDataset:
        try:
            from CGNS import MAP  # pyCGNS
        except ImportError as exc:
            raise ImportError("pyCGNS(CGNS.MAP) 미설치") from exc

        logger.debug("CGNS.MAP 로 읽기: %s", path)
        tree, _links, _paths = MAP.load(str(path))
        return _cgns_tree_to_cfd_dataset(tree, str(path))

    def _read_with_h5py(self, path: Path) -> CFDDataset:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py 미설치") from exc

        logger.debug("h5py 로 CGNS 읽기: %s", path)
        with h5py.File(str(path), "r") as f:
            return _h5py_cgns_to_cfd_dataset(f, str(path))

    def _read_with_meshio(self, path: Path) -> CFDDataset:
        try:
            import meshio
        except ImportError as exc:
            raise ImportError("meshio 미설치") from exc

        logger.debug("meshio 로 CGNS 읽기: %s", path)
        mesh = meshio.read(str(path), file_format="cgns")
        return meshio_to_cfd_dataset(mesh, str(path), "meshio/CGNS")


# ---------------------------------------------------------------------------
# pyCGNS tree → CFDDataset
# ---------------------------------------------------------------------------


def _cgns_tree_to_cfd_dataset(tree: Any, source_file: str = "") -> CFDDataset:
    """pyCGNS tree 구조에서 CFDDataset 을 구성한다.

    CGNS 트리 구조: [name, value, children, type]
    Base → Zone → GridCoordinates/FlowSolution 순으로 탐색한다.

    Args:
        tree: CGNS.MAP.load() 가 반환한 CGNS 트리.
        source_file: 원본 파일 경로.

    Returns:
        CFDDataset.
    """
    import numpy as np

    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("pyvista 가 필요합니다") from exc

    nodes_x: list[Any] = []
    nodes_y: list[Any] = []
    nodes_z: list[Any] = []
    field_data: dict[str, Any] = {}

    def _traverse(node: Any) -> None:
        """CGNS 트리를 재귀 탐색한다."""
        if not isinstance(node, (list, tuple)) or len(node) < 4:
            return
        name, value, children, ntype = node[0], node[1], node[2], node[3]

        if ntype == "GridCoordinates_t":
            for child in children:
                cname, cval = child[0], child[1]
                if cval is None:
                    continue
                arr = np.asarray(cval).ravel()
                if "CoordinateX" in cname:
                    nodes_x.append(arr)
                elif "CoordinateY" in cname:
                    nodes_y.append(arr)
                elif "CoordinateZ" in cname:
                    nodes_z.append(arr)

        elif ntype == "FlowSolution_t":
            for child in children:
                cname, cval = child[0], child[1]
                if cval is not None:
                    field_data[cname] = np.asarray(cval).ravel()

        for child in children:
            _traverse(child)

    _traverse(tree)

    if not nodes_x:
        raise ValueError("CGNS 트리에서 GridCoordinates 를 찾을 수 없습니다.")

    x = np.concatenate(nodes_x)
    y = np.concatenate(nodes_y) if nodes_y else np.zeros_like(x)
    z = np.concatenate(nodes_z) if nodes_z else np.zeros_like(x)
    points = np.column_stack([x, y, z])

    mesh = pv.PolyData(points).cast_to_unstructured_grid()
    for fname, arr in field_data.items():
        if len(arr) == len(points):
            mesh.point_data[fname] = arr

    field_names = sorted(field_data.keys())
    logger.debug(
        "pyCGNS → CFDDataset: %d 노드, 필드=%s", len(points), field_names
    )

    return CFDDataset(
        mesh=mesh,
        time_steps=[0.0],
        field_names=field_names,
        metadata={"reader": "pyCGNS (CGNS.MAP)", "source_file": source_file},
    )


# ---------------------------------------------------------------------------
# h5py CGNS → CFDDataset
# ---------------------------------------------------------------------------


def _h5py_cgns_to_cfd_dataset(f: Any, source_file: str = "") -> CFDDataset:
    """h5py 로 열린 CGNS HDF5 파일에서 CFDDataset 을 구성한다.

    CGNS HDF5 레이아웃:
        /Base/Zone/GridCoordinates/CoordinateX|Y|Z
        /Base/Zone/FlowSolution/<field>

    Args:
        f: h5py.File 핸들 (읽기 모드).
        source_file: 원본 파일 경로.

    Returns:
        CFDDataset.
    """
    import numpy as np

    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("pyvista 가 필요합니다") from exc

    coords_x: list[Any] = []
    coords_y: list[Any] = []
    coords_z: list[Any] = []
    field_data: dict[str, Any] = {}

    def _walk(group: Any) -> None:
        for key in group.keys():
            item = group[key]
            if hasattr(item, "keys"):
                # 그룹
                label = item.attrs.get("label", b"").decode("utf-8", errors="replace")
                if label in ("GridCoordinates_t",) or "GridCoordinates" in key:
                    for coord_key in item.keys():
                        coord_item = item[coord_key]
                        if not hasattr(coord_item, "keys"):
                            arr = np.asarray(coord_item).ravel()
                            if "X" in coord_key:
                                coords_x.append(arr)
                            elif "Y" in coord_key:
                                coords_y.append(arr)
                            elif "Z" in coord_key:
                                coords_z.append(arr)
                        elif " data" in coord_item:
                            arr = np.asarray(coord_item[" data"]).ravel()
                            if "X" in coord_key:
                                coords_x.append(arr)
                            elif "Y" in coord_key:
                                coords_y.append(arr)
                            elif "Z" in coord_key:
                                coords_z.append(arr)
                elif label in ("FlowSolution_t",) or "FlowSolution" in key:
                    for field_key in item.keys():
                        field_item = item[field_key]
                        try:
                            if hasattr(field_item, "keys") and " data" in field_item:
                                arr = np.asarray(field_item[" data"]).ravel()
                            elif not hasattr(field_item, "keys"):
                                arr = np.asarray(field_item).ravel()
                            else:
                                continue
                            field_data[field_key] = arr
                        except Exception:
                            pass
                else:
                    _walk(item)

    _walk(f)

    if not coords_x:
        # 단순 폴백: 최상위 데이터셋에서 좌표 추출 시도
        raise ValueError("h5py CGNS 파싱: GridCoordinates 를 찾을 수 없습니다.")

    x = np.concatenate(coords_x)
    y = np.concatenate(coords_y) if coords_y else np.zeros_like(x)
    z = np.concatenate(coords_z) if coords_z else np.zeros_like(x)
    points = np.column_stack([x, y, z])

    mesh = pv.PolyData(points).cast_to_unstructured_grid()
    for fname, arr in field_data.items():
        if len(arr) == len(points):
            mesh.point_data[fname] = arr

    field_names = sorted(field_data.keys())
    logger.debug(
        "h5py CGNS → CFDDataset: %d 노드, 필드=%s", len(points), field_names
    )

    return CFDDataset(
        mesh=mesh,
        time_steps=[0.0],
        field_names=field_names,
        metadata={"reader": "h5py/CGNS", "source_file": source_file},
    )
