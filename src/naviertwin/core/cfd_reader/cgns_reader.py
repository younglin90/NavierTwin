"""CGNS (.cgns) 파일 리더 모듈.

CGNS (CFD General Notation System) HDF5 파일을 읽는다.

폴백 체인:
    1. pyvista.CGNSReader (vtkCGNSReader)
    2. pyCGNS — CGNS.MAP.load()
    3. h5py — HDF5 직접 파싱
    4. meshio

확장:
    - pyCGNS/h5py 폴백 경로에서 ``Elements_t`` 셀 연결성을 파싱해
      점 구름이 아닌 진짜 UnstructuredGrid 를 구성한다.
    - ``ZoneBC_t`` 를 파싱해 OpenFOAM 리더와 동일한 계약으로
      ``metadata["boundary_patches"]`` / ``metadata["boundary_patch_meshes"]``
      를 채우고, BCWall* 계열 patch 는 ``metadata["auto_wall_patches"]`` 에
      자동 분류한다.
    - MIXED / NGON_n / NFACE_n 과 다중 존 연결성을 지원한다.

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

# ---------------------------------------------------------------------------
# CGNS ElementType 코드 → (VTK celltype, 요소당 노드 수)
# CGNS SIDS: NODE=2, BAR_2=3, TRI_3=5, QUAD_4=7, TETRA_4=10,
#            PYRA_5=12, PENTA_6=14, HEXA_8=17
# VTK:       VERTEX=1, LINE=3, TRIANGLE=5, QUAD=9, TETRA=10,
#            HEXAHEDRON=12, WEDGE=13, PYRAMID=14
# ---------------------------------------------------------------------------
_CGNS_ELEM_TO_VTK: dict[int, tuple[int, int]] = {
    2: (1, 1),  # NODE → VTK_VERTEX
    3: (3, 2),  # BAR_2 → VTK_LINE
    5: (5, 3),  # TRI_3 → VTK_TRIANGLE
    7: (9, 4),  # QUAD_4 → VTK_QUAD
    10: (10, 4),  # TETRA_4 → VTK_TETRA
    12: (14, 5),  # PYRA_5 → VTK_PYRAMID
    14: (13, 6),  # PENTA_6 → VTK_WEDGE
    17: (12, 8),  # HEXA_8 → VTK_HEXAHEDRON
}

_CGNS_VARIABLE_ELEM = {20: "MIXED", 22: "NGON_n", 23: "NFACE_n"}

#: BC 타입 문자열이 이 접두사로 시작하면 wall 로 분류한다
#: (BCWall, BCWallViscous, BCWallViscousHeatFlux, BCWallViscousIsothermal,
#:  BCWallInviscid 등).
_WALL_BC_PREFIX = "bcwall"


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
        dataset = pyvista_to_cfd_dataset(mesh, str(path), "pyvista.CGNSReader")

        # vtkCGNSReader 는 ZoneBC 정보를 patch 메타데이터로 노출하지 않는다.
        # h5py 로 ZoneBC_t 만 후처리 파싱해 patch 이름/타입/wall 여부를 붙인다.
        # (병합된 MultiBlock 의 점 번호와 zone-local PointList 번호가 일치한다는
        #  보장이 없으므로 여기서는 표면 서브메쉬는 만들지 않는다 — 메타만.)
        try:
            patches = _parse_zonebc_via_h5py(path)
        except Exception as e:  # noqa: BLE001 — 메타 후처리 실패는 치명적이지 않다
            logger.debug("pyvista 경로 ZoneBC 후처리 실패: %s", e)
            patches = []
        if patches:
            _attach_boundary_metadata(
                dataset.metadata, dataset.mesh, patches, extract_meshes=False
            )
        return dataset

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
# 공통 헬퍼 — 문자열/인덱스 디코딩
# ---------------------------------------------------------------------------


def _decode_cgns_string(value: Any) -> str:
    """CGNS 노드 값(바이트/문자 배열)을 파이썬 문자열로 디코드한다.

    pyCGNS 는 문자열을 ``dtype='S1'`` 또는 int8 배열로, h5py 는 바이트
    데이터셋으로 저장한다. 널 문자와 공백을 제거한 문자열을 돌려준다.

    Args:
        value: CGNS 노드 값 (numpy 배열, bytes, str 등).

    Returns:
        디코드된 문자열. 실패 시 빈 문자열.
    """
    import numpy as np

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00").strip()
    try:
        arr = np.asarray(value)
        return arr.tobytes().decode("utf-8", errors="replace").strip("\x00").strip()
    except Exception:  # noqa: BLE001 — 알 수 없는 값 타입 보호
        return ""


def _point_indices_from_bc(
    point_list: Any, point_range: Any
) -> Any | None:
    """BC 의 PointList/PointRange 를 0-based 점 인덱스 배열로 변환한다.

    Args:
        point_list: PointList 값 (1-based 인덱스 배열) 또는 None.
        point_range: PointRange 값 (``[begin, end]``, 1-based) 또는 None.

    Returns:
        0-based int64 인덱스 ndarray. 정보가 없으면 None.
    """
    import numpy as np

    if point_list is not None:
        idx = np.asarray(point_list).ravel().astype(np.int64)
        if idx.size == 0:
            return None
        return idx - 1
    if point_range is not None:
        rng = np.asarray(point_range).ravel().astype(np.int64)
        if rng.size >= 2:
            begin, end = int(rng[0]), int(rng[1])
            if end >= begin:
                return np.arange(begin - 1, end, dtype=np.int64)
    return None


# ---------------------------------------------------------------------------
# 공통 헬퍼 — Elements_t 섹션 → VTK 셀, 메쉬/메타데이터 구성
# ---------------------------------------------------------------------------


def _sections_to_vtk_cells(
    sections: list[dict[str, Any]], n_points: int
) -> tuple[Any, Any] | None:
    """Elements_t 섹션 목록을 VTK 셀 배열로 변환한다.

    Args:
        sections: ``{"name", "etype", "conn"(0-based ravel), "range"}`` 목록.
        n_points: 존 전체 점 개수 (연결성 범위 검증용).

    Returns:
        ``(cells, celltypes)`` 튜플. 변환 가능한 섹션이 없으면 None.
    """
    import numpy as np

    cell_records: list[Any] = []
    cell_types: list[int] = []

    def _sort_key(sec: dict[str, Any]) -> int:
        rng = sec.get("range")
        if rng is None:
            return 0
        flat = np.asarray(rng).ravel()
        return int(flat[0]) if flat.size else 0

    ordered = sorted(sections, key=lambda sec: (str(sec.get("zone", "")), _sort_key(sec)))

    def add_cell(nodes: Any, vtk_type: int) -> bool:
        node_ids = np.asarray(nodes, dtype=np.int64).ravel()
        if node_ids.size == 0 or node_ids.min() < 0 or node_ids.max() >= n_points:
            return False
        cell_records.append(np.concatenate(([node_ids.size], node_ids)))
        cell_types.append(vtk_type)
        return True

    def split_variable(sec: dict[str, Any]) -> list[Any]:
        conn = np.asarray(sec["conn"]).ravel().astype(np.int64)
        offsets = np.asarray(sec.get("offsets", ())).ravel().astype(np.int64)
        if offsets.size < 2:
            logger.warning(
                "CGNS Elements_t '%s': ElementStartOffset 누락 — 건너뜁니다.",
                sec.get("name", "?"),
            )
            return []
        if offsets[0] == 1 and offsets[-1] == conn.size + 1:
            offsets = offsets - 1
        if offsets[0] != 0 or offsets[-1] != conn.size or np.any(np.diff(offsets) < 0):
            logger.warning(
                "CGNS Elements_t '%s': 잘못된 ElementStartOffset — 건너뜁니다.",
                sec.get("name", "?"),
            )
            return []
        return [conn[start:end] for start, end in zip(offsets[:-1], offsets[1:], strict=True)]

    # NGON faces are indexed by ElementRange and referenced by signed NFACE ids.
    face_maps: dict[tuple[str, int], Any] = {}
    zones_with_nface = {
        str(sec.get("zone", "")) for sec in ordered if int(sec["etype"]) == 23
    }
    for sec in ordered:
        if int(sec["etype"]) != 22:
            continue
        zone = str(sec.get("zone", ""))
        point_offset = int(sec.get("point_offset", 0))
        one_based = bool(sec.get("one_based", False))
        chunks = split_variable(sec)
        element_range = np.asarray(sec.get("range", ())).ravel()
        first_id = int(element_range[0]) if element_range.size else 1
        for index, chunk in enumerate(chunks):
            nodes = chunk - 1 if one_based else chunk
            nodes = nodes + point_offset
            if nodes.size < 3 or nodes.min() < 0 or nodes.max() >= n_points:
                logger.warning("CGNS NGON '%s': 잘못된 face %d", sec.get("name", "?"), index)
                continue
            face_maps[(zone, first_id + index)] = nodes
            if zone not in zones_with_nface:
                add_cell(nodes, 7)  # VTK_POLYGON

    for sec in ordered:
        name = sec.get("name", "?")
        etype = int(sec["etype"])
        if etype == 22:
            continue
        point_offset = int(sec.get("point_offset", 0))
        one_based = bool(sec.get("one_based", False))
        if etype == 20:
            conn = np.asarray(sec["conn"]).ravel().astype(np.int64)
            cursor = 0
            while cursor < conn.size:
                mixed_type = int(conn[cursor])
                cursor += 1
                mapping = _CGNS_ELEM_TO_VTK.get(mixed_type)
                if mapping is None:
                    logger.warning(
                        "CGNS MIXED '%s': ElementType %d 미지원 — 나머지 섹션 중단",
                        name,
                        mixed_type,
                    )
                    break
                vtk_type, node_count = mapping
                if cursor + node_count > conn.size:
                    logger.warning("CGNS MIXED '%s': 잘린 연결성", name)
                    break
                nodes = conn[cursor : cursor + node_count]
                cursor += node_count
                nodes = (nodes - 1 if one_based else nodes) + point_offset
                if not add_cell(nodes, vtk_type):
                    logger.warning("CGNS MIXED '%s': 범위 밖 연결성", name)
            continue
        if etype == 23:
            zone = str(sec.get("zone", ""))
            for references in split_variable(sec):
                faces = [face_maps.get((zone, abs(int(ref)))) for ref in references]
                if not faces or any(face is None for face in faces):
                    logger.warning("CGNS NFACE '%s': 참조 NGON 누락", name)
                    continue
                face_stream = [len(faces)]
                for face in faces:
                    face_stream.extend([len(face), *face.tolist()])
                cell_records.append(
                    np.asarray([len(face_stream), *face_stream], dtype=np.int64)
                )
                cell_types.append(42)  # VTK_POLYHEDRON
            continue
        mapping = _CGNS_ELEM_TO_VTK.get(etype)
        if mapping is None:
            logger.warning(
                "CGNS Elements_t '%s': 알 수 없는 ElementType 코드 %d — 건너뜁니다.",
                name,
                etype,
            )
            continue
        vtk_type, nodes_per_elem = mapping
        conn = np.asarray(sec["conn"]).ravel().astype(np.int64)
        if one_based:
            conn = conn - 1
        conn = conn + point_offset
        if conn.size == 0 or conn.size % nodes_per_elem != 0:
            logger.warning(
                "CGNS Elements_t '%s': 연결성 길이 %d 가 요소당 노드 수 %d 와 "
                "맞지 않음 — 건너뜁니다.",
                name,
                conn.size,
                nodes_per_elem,
            )
            continue
        if conn.min() < 0 or conn.max() >= n_points:
            logger.warning(
                "CGNS Elements_t '%s': 연결성 인덱스가 점 범위 [0, %d) 를 벗어남 — "
                "건너뜁니다.",
                name,
                n_points,
            )
            continue
        for nodes in conn.reshape(-1, nodes_per_elem):
            add_cell(nodes, vtk_type)

    if not cell_records:
        return None
    return np.concatenate(cell_records), np.asarray(cell_types, dtype=np.uint8)


def _build_mesh_from_parts(
    points: Any, sections: list[dict[str, Any]], field_data: dict[str, Any]
) -> Any:
    """점 좌표 + Elements_t 섹션 + 필드에서 pyvista UnstructuredGrid 를 만든다.

    연결성이 하나도 없으면 기존 동작(점 구름)으로 폴백한다.

    Args:
        points: (N, 3) 점 좌표 배열.
        sections: Elements_t 섹션 목록 (없으면 빈 리스트).
        field_data: {필드 이름: 1D 배열}.

    Returns:
        pyvista.UnstructuredGrid.
    """
    import pyvista as pv

    cells_and_types = (
        _sections_to_vtk_cells(sections, len(points)) if sections else None
    )
    if cells_and_types is not None:
        cells, celltypes = cells_and_types
        mesh = pv.UnstructuredGrid(cells, celltypes, points)
    else:
        if sections:
            logger.warning(
                "CGNS: 변환 가능한 Elements_t 섹션이 없어 점 구름으로 폴백합니다."
            )
        mesh = pv.PolyData(points).cast_to_unstructured_grid()

    for fname, arr in field_data.items():
        if len(arr) == mesh.n_points:
            mesh.point_data[fname] = arr
        elif len(arr) == mesh.n_cells:
            mesh.cell_data[fname] = arr
    return mesh


def _attach_boundary_metadata(
    metadata: dict[str, Any],
    mesh: Any,
    patches: list[dict[str, Any]],
    *,
    extract_meshes: bool = True,
) -> None:
    """ZoneBC 패치 정보를 OpenFOAM 리더와 동일한 계약으로 metadata 에 붙인다.

    계약 (openfoam_reader._extract_boundary_patches 참조):
        - ``metadata["boundary_patches"]`` =
          ``{이름: {"n_cells": int, "n_points": int, "type": str, "is_wall": bool}}``
        - ``metadata["boundary_patch_meshes"]`` = ``{이름: PolyData 표면}``
          (서브메쉬 구성이 가능한 patch 만)
        - ``metadata["auto_wall_patches"]`` = wall 로 분류된 patch 이름 목록

    Args:
        metadata: 대상 CFDDataset.metadata 딕셔너리 (in-place 수정).
        mesh: 전체 볼륨 메쉬 (표면 서브메쉬 추출용).
        patches: ``{"name", "type", "point_indices", "grid_location"}`` 목록.
        extract_meshes: False 이면 표면 서브메쉬 추출을 생략한다
            (pyvista 경로처럼 점 번호 대응이 보장되지 않는 경우).
    """
    import numpy as np

    if not patches:
        return

    boundary_patches: dict[str, dict[str, Any]] = {}
    patch_meshes: dict[str, Any] = {}
    wall_names: list[str] = []

    for patch in patches:
        name = str(patch["name"])
        bc_type = str(patch.get("type", ""))
        is_wall = bc_type.lower().startswith(_WALL_BC_PREFIX)
        grid_location = str(patch.get("grid_location") or "Vertex")
        indices = patch.get("point_indices")

        n_patch_points = 0
        sub_mesh = None
        if indices is not None:
            indices = np.asarray(indices).ravel().astype(np.int64)
            n_patch_points = int(indices.size)
            # PointList 가 점(Vertex) 위치일 때만 점 인덱스로 표면을 만들 수 있다.
            if (
                extract_meshes
                and n_patch_points > 0
                and grid_location in ("Vertex", "")
            ):
                sub_mesh = _extract_patch_surface(mesh, indices, name)

        entry: dict[str, Any] = {
            "n_cells": int(sub_mesh.n_cells) if sub_mesh is not None else 0,
            "n_points": (
                int(sub_mesh.n_points) if sub_mesh is not None else n_patch_points
            ),
            "type": bc_type,
            "is_wall": is_wall,
        }
        boundary_patches[name] = entry
        if sub_mesh is not None:
            patch_meshes[name] = sub_mesh
        if is_wall:
            wall_names.append(name)

    metadata["boundary_patches"] = boundary_patches
    if patch_meshes:
        metadata["boundary_patch_meshes"] = patch_meshes
    metadata["auto_wall_patches"] = sorted(wall_names)

    if wall_names:
        logger.info(
            "CGNS ZoneBC: patch %d개, wall 자동 인식 %d개: %s",
            len(boundary_patches),
            len(wall_names),
            sorted(wall_names),
        )
    else:
        logger.info("CGNS ZoneBC: patch %d개 (wall 없음)", len(boundary_patches))


def _extract_patch_surface(mesh: Any, indices: Any, name: str) -> Any | None:
    """0-based 점 인덱스로 patch 표면(PolyData)을 추출한다.

    점 인덱스에 완전히 포함되는 셀이 있으면 그 표면을, 없으면 해당 점들의
    점 구름 PolyData 를 돌려준다 (다운스트림 벽면 거리 계산 등에 사용 가능).

    Args:
        mesh: 전체 볼륨 UnstructuredGrid.
        indices: 0-based 점 인덱스 배열.
        name: patch 이름 (로그용).

    Returns:
        pyvista.PolyData 또는 None (추출 실패).
    """
    import numpy as np

    try:
        import pyvista as pv
    except ImportError:
        return None

    try:
        valid = indices[(indices >= 0) & (indices < mesh.n_points)]
        if valid.size == 0:
            return None
        if valid.size < indices.size:
            logger.warning(
                "CGNS patch '%s': PointList 인덱스 %d개가 점 범위를 벗어나 제외",
                name,
                int(indices.size - valid.size),
            )
        if mesh.n_cells > 0:
            sub = mesh.extract_points(
                valid, adjacent_cells=False, include_cells=True
            )
            if sub is not None and sub.n_cells > 0:
                surf = sub.extract_surface()
                if surf.n_points > 0:
                    return surf
        return pv.PolyData(np.asarray(mesh.points)[valid])
    except Exception as e:  # noqa: BLE001 — patch 하나 실패가 전체를 막지 않게
        logger.debug("CGNS patch '%s' 표면 추출 실패: %s", name, e)
        return None


def _pycgns_zone_nodes(tree: Any) -> list[Any]:
    """Return every Zone_t node without descending into a matched zone."""
    zones: list[Any] = []

    def visit(node: Any) -> None:
        if not isinstance(node, (list, tuple)) or len(node) < 4:
            return
        if node[3] == "Zone_t":
            zones.append(node)
            return
        for child in node[2]:
            visit(child)

    visit(tree)
    return zones


def _h5py_zone_groups(root: Any) -> list[tuple[str, Any]]:
    """Return named Zone_t groups below an HDF5 root."""
    zones: list[tuple[str, Any]] = []

    def visit(group: Any) -> None:
        for key in group.keys():
            item = group[key]
            if not hasattr(item, "keys"):
                continue
            if _h5py_label(item) == "Zone_t":
                zones.append((str(key), item))
            else:
                visit(item)

    visit(root)
    return zones


def _combine_zone_datasets(
    zone_names: list[str],
    datasets: list[CFDDataset],
    *,
    reader: str,
    source_file: str,
) -> CFDDataset:
    """Merge independently parsed zones while preserving local connectivity."""
    import pyvista as pv

    blocks = pv.MultiBlock([dataset.mesh for dataset in datasets])
    mesh = blocks.combine(merge_points=False)
    boundary_patches: dict[str, Any] = {}
    patch_meshes: dict[str, Any] = {}
    wall_names: list[str] = []
    for zone_name, dataset in zip(zone_names, datasets, strict=True):
        metadata = dataset.metadata
        for patch_name, patch in metadata.get("boundary_patches", {}).items():
            qualified = f"{zone_name}/{patch_name}"
            boundary_patches[qualified] = patch
            if patch.get("is_wall"):
                wall_names.append(qualified)
        for patch_name, patch_mesh in metadata.get(
            "boundary_patch_meshes", {}
        ).items():
            patch_meshes[f"{zone_name}/{patch_name}"] = patch_mesh

    metadata: dict[str, Any] = {
        "reader": reader,
        "source_file": source_file,
        "zone_count": len(zone_names),
        "zone_names": list(zone_names),
    }
    if boundary_patches:
        metadata["boundary_patches"] = boundary_patches
        metadata["auto_wall_patches"] = sorted(wall_names)
    if patch_meshes:
        metadata["boundary_patch_meshes"] = patch_meshes
    field_names = sorted({*mesh.point_data.keys(), *mesh.cell_data.keys()})
    logger.info(
        "CGNS 다중 존 병합: zones=%d, points=%d, cells=%d",
        len(zone_names),
        mesh.n_points,
        mesh.n_cells,
    )
    return CFDDataset(
        mesh=mesh,
        time_steps=[0.0],
        field_names=field_names,
        metadata=metadata,
    )

# ---------------------------------------------------------------------------
# pyCGNS tree → CFDDataset
# ---------------------------------------------------------------------------


def _cgns_tree_to_cfd_dataset(tree: Any, source_file: str = "") -> CFDDataset:
    """pyCGNS tree 구조에서 CFDDataset 을 구성한다.

    CGNS 트리 구조: [name, value, children, type]
    Base → Zone → GridCoordinates/Elements_t/ZoneBC_t/FlowSolution 순으로
    탐색한다. Elements_t 가 있으면 셀 연결성을 가진 UnstructuredGrid 를,
    없으면 기존처럼 점 구름을 만든다.

    Args:
        tree: CGNS.MAP.load() 가 반환한 CGNS 트리.
        source_file: 원본 파일 경로.

    Returns:
        CFDDataset.
    """
    import numpy as np

    try:
        import pyvista  # noqa: F401  (availability probe)
    except ImportError as exc:
        raise ImportError("pyvista 가 필요합니다") from exc

    zone_nodes = _pycgns_zone_nodes(tree)
    if len(zone_nodes) > 1:
        zone_names = [str(zone[0]) for zone in zone_nodes]
        zone_datasets = [
            _cgns_tree_to_cfd_dataset(zone, source_file) for zone in zone_nodes
        ]
        return _combine_zone_datasets(
            zone_names,
            zone_datasets,
            reader="pyCGNS (CGNS.MAP)",
            source_file=source_file,
        )

    nodes_x: list[Any] = []
    nodes_y: list[Any] = []
    nodes_z: list[Any] = []
    field_data: dict[str, Any] = {}
    sections: list[dict[str, Any]] = []
    bc_patches: list[dict[str, Any]] = []

    def _child_by_type(children: list[Any], ntype: str) -> Any | None:
        for child in children:
            if isinstance(child, (list, tuple)) and len(child) >= 4 and child[3] == ntype:
                return child
        return None

    def _child_by_name(children: list[Any], cname: str) -> Any | None:
        for child in children:
            if isinstance(child, (list, tuple)) and len(child) >= 4 and child[0] == cname:
                return child
        return None

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
                cname, cval, _cchildren, ctype = child[0], child[1], child[2], child[3]
                if cval is not None and ctype == "DataArray_t":
                    field_data[cname] = np.asarray(cval).ravel()

        elif ntype == "Elements_t":
            etype = None
            if value is not None:
                vals = np.asarray(value).ravel()
                if vals.size >= 1:
                    etype = int(vals[0])
            conn_node = _child_by_name(children, "ElementConnectivity")
            range_node = _child_by_name(children, "ElementRange")
            offsets_node = _child_by_name(children, "ElementStartOffset")
            if etype is not None and conn_node is not None and conn_node[1] is not None:
                sections.append(
                    {
                        "name": name,
                        "etype": etype,
                        "conn": np.asarray(conn_node[1]).ravel().astype(np.int64),
                        "one_based": True,
                        "offsets": (
                            offsets_node[1] if offsets_node is not None else None
                        ),
                        "range": (
                            range_node[1] if range_node is not None else None
                        ),
                    }
                )

        elif ntype == "ZoneBC_t":
            for child in children:
                if not isinstance(child, (list, tuple)) or len(child) < 4:
                    continue
                if child[3] != "BC_t":
                    continue
                bc_name, bc_value, bc_children = child[0], child[1], child[2]
                point_list_node = _child_by_name(bc_children, "PointList")
                point_range_node = _child_by_name(bc_children, "PointRange")
                loc_node = _child_by_type(bc_children, "GridLocation_t")
                bc_patches.append(
                    {
                        "name": bc_name,
                        "type": _decode_cgns_string(bc_value),
                        "point_indices": _point_indices_from_bc(
                            point_list_node[1] if point_list_node is not None else None,
                            point_range_node[1]
                            if point_range_node is not None
                            else None,
                        ),
                        "grid_location": _decode_cgns_string(
                            loc_node[1] if loc_node is not None else None
                        ),
                    }
                )
            return  # BC_t 하위는 별도 재귀 불필요

        for child in children:
            _traverse(child)

    _traverse(tree)

    if not nodes_x:
        raise ValueError("CGNS 트리에서 GridCoordinates 를 찾을 수 없습니다.")

    x = np.concatenate(nodes_x)
    y = np.concatenate(nodes_y) if nodes_y else np.zeros_like(x)
    z = np.concatenate(nodes_z) if nodes_z else np.zeros_like(x)
    points = np.column_stack([x, y, z])

    mesh = _build_mesh_from_parts(points, sections, field_data)

    metadata: dict[str, Any] = {
        "reader": "pyCGNS (CGNS.MAP)",
        "source_file": source_file,
    }
    _attach_boundary_metadata(metadata, mesh, bc_patches)

    field_names = sorted(field_data.keys())
    logger.debug(
        "pyCGNS → CFDDataset: %d 노드, %d 셀, 필드=%s",
        mesh.n_points,
        mesh.n_cells,
        field_names,
    )

    return CFDDataset(
        mesh=mesh,
        time_steps=[0.0],
        field_names=field_names,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# h5py CGNS → CFDDataset
# ---------------------------------------------------------------------------


def _h5py_read_value(group: Any) -> Any | None:
    """CGNS HDF5 노드 그룹에서 ``' data'`` 데이터셋 값을 읽는다."""
    import numpy as np

    if hasattr(group, "keys") and " data" in group:
        return np.asarray(group[" data"])
    if not hasattr(group, "keys"):
        return np.asarray(group)
    return None


def _h5py_label(item: Any) -> str:
    """h5py 그룹의 CGNS label 어트리뷰트를 문자열로 읽는다."""
    raw = item.attrs.get("label", b"")
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace").strip("\x00").strip()
    return str(raw).strip()


def _h5py_parse_elements(item: Any, name: str) -> dict[str, Any] | None:
    """label 이 ``Elements_t`` 인 h5py 그룹에서 섹션 정보를 파싱한다."""
    import numpy as np

    value = _h5py_read_value(item)
    if value is None:
        return None
    vals = np.asarray(value).ravel()
    if vals.size < 1:
        return None
    etype = int(vals[0])

    conn = None
    if "ElementConnectivity" in item:
        conn = _h5py_read_value(item["ElementConnectivity"])
    if conn is None:
        return None

    erange = None
    if "ElementRange" in item:
        erange = _h5py_read_value(item["ElementRange"])
    offsets = None
    if "ElementStartOffset" in item:
        offsets = _h5py_read_value(item["ElementStartOffset"])

    return {
        "name": name,
        "etype": etype,
        "conn": np.asarray(conn).ravel().astype(np.int64),
        "one_based": True,
        "offsets": offsets,
        "range": erange,
    }


def _h5py_parse_zonebc(item: Any) -> list[dict[str, Any]]:
    """label 이 ``ZoneBC_t`` 인 h5py 그룹에서 BC patch 목록을 파싱한다."""
    patches: list[dict[str, Any]] = []
    for bc_name in item.keys():
        bc_group = item[bc_name]
        if not hasattr(bc_group, "keys"):
            continue
        if _h5py_label(bc_group) != "BC_t":
            continue
        bc_type = _decode_cgns_string(_h5py_read_value(bc_group))
        point_list = None
        point_range = None
        if "PointList" in bc_group:
            point_list = _h5py_read_value(bc_group["PointList"])
        if "PointRange" in bc_group:
            point_range = _h5py_read_value(bc_group["PointRange"])
        grid_location = ""
        if "GridLocation" in bc_group:
            grid_location = _decode_cgns_string(
                _h5py_read_value(bc_group["GridLocation"])
            )
        patches.append(
            {
                "name": str(bc_name),
                "type": bc_type,
                "point_indices": _point_indices_from_bc(point_list, point_range),
                "grid_location": grid_location,
            }
        )
    return patches


def _parse_zonebc_via_h5py(path: Path) -> list[dict[str, Any]]:
    """CGNS 파일에서 ZoneBC_t patch 목록만 h5py 로 파싱한다.

    pyvista 1차 경로의 후처리용 — vtkCGNSReader 가 노출하지 않는
    patch 이름/타입 메타를 보강한다.

    Args:
        path: .cgns 파일 경로.

    Returns:
        patch 딕셔너리 목록 (없으면 빈 리스트).
    """
    import h5py

    patches: list[dict[str, Any]] = []

    def _walk(group: Any) -> None:
        for key in group.keys():
            item = group[key]
            if not hasattr(item, "keys"):
                continue
            if _h5py_label(item) == "ZoneBC_t":
                patches.extend(_h5py_parse_zonebc(item))
            else:
                _walk(item)

    with h5py.File(str(path), "r") as f:
        _walk(f)
    return patches


def _h5py_cgns_to_cfd_dataset(f: Any, source_file: str = "") -> CFDDataset:
    """h5py 로 열린 CGNS HDF5 파일에서 CFDDataset 을 구성한다.

    CGNS HDF5 레이아웃:
        /Base/Zone/GridCoordinates/CoordinateX|Y|Z
        /Base/Zone/<Elements_t 그룹>/ElementConnectivity|ElementRange
        /Base/Zone/ZoneBC/<BC_t 그룹>/PointList|PointRange|GridLocation
        /Base/Zone/FlowSolution/<field>

    Elements_t 가 있으면 셀 연결성을 가진 UnstructuredGrid 를, 없으면
    기존처럼 점 구름을 만든다.

    Args:
        f: h5py.File 핸들 (읽기 모드).
        source_file: 원본 파일 경로.

    Returns:
        CFDDataset.
    """
    import numpy as np

    try:
        import pyvista  # noqa: F401  (availability probe)
    except ImportError as exc:
        raise ImportError("pyvista 가 필요합니다") from exc

    zone_groups = _h5py_zone_groups(f)
    if len(zone_groups) > 1:
        zone_names = [name for name, _group in zone_groups]
        zone_datasets = [
            _h5py_cgns_to_cfd_dataset(group, source_file)
            for _name, group in zone_groups
        ]
        return _combine_zone_datasets(
            zone_names,
            zone_datasets,
            reader="h5py/CGNS",
            source_file=source_file,
        )

    coords_x: list[Any] = []
    coords_y: list[Any] = []
    coords_z: list[Any] = []
    field_data: dict[str, Any] = {}
    sections: list[dict[str, Any]] = []
    bc_patches: list[dict[str, Any]] = []

    def _walk(group: Any) -> None:
        for key in group.keys():
            item = group[key]
            if not hasattr(item, "keys"):
                continue
            label = _h5py_label(item)
            if label == "GridCoordinates_t" or "GridCoordinates" in key:
                for coord_key in item.keys():
                    coord_item = item[coord_key]
                    arr_val = _h5py_read_value(coord_item)
                    if arr_val is None:
                        continue
                    arr = np.asarray(arr_val).ravel()
                    if "X" in coord_key:
                        coords_x.append(arr)
                    elif "Y" in coord_key:
                        coords_y.append(arr)
                    elif "Z" in coord_key:
                        coords_z.append(arr)
            elif label == "FlowSolution_t" or "FlowSolution" in key:
                for field_key in item.keys():
                    try:
                        arr_val = _h5py_read_value(item[field_key])
                        if arr_val is None:
                            continue
                        field_data[field_key] = np.asarray(arr_val).ravel()
                    except Exception:  # noqa: BLE001 — 비정상 필드 노드 보호
                        pass
            elif label == "Elements_t":
                try:
                    section = _h5py_parse_elements(item, str(key))
                except Exception as e:  # noqa: BLE001 — 비정상 섹션 보호
                    logger.warning("CGNS Elements_t '%s' 파싱 실패: %s", key, e)
                    section = None
                if section is not None:
                    sections.append(section)
            elif label == "ZoneBC_t":
                try:
                    bc_patches.extend(_h5py_parse_zonebc(item))
                except Exception as e:  # noqa: BLE001 — 비정상 ZoneBC 보호
                    logger.warning("CGNS ZoneBC_t '%s' 파싱 실패: %s", key, e)
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

    mesh = _build_mesh_from_parts(points, sections, field_data)

    metadata: dict[str, Any] = {"reader": "h5py/CGNS", "source_file": source_file}
    _attach_boundary_metadata(metadata, mesh, bc_patches)

    field_names = sorted(field_data.keys())
    logger.debug(
        "h5py CGNS → CFDDataset: %d 노드, %d 셀, 필드=%s",
        mesh.n_points,
        mesh.n_cells,
        field_names,
    )

    return CFDDataset(
        mesh=mesh,
        time_steps=[0.0],
        field_names=field_names,
        metadata=metadata,
    )
