"""NavierTwin v5.1 — CGNS 셀 연결성 + ZoneBC 자동 wall 인식 테스트.

pyCGNS tree 폴백 경로와 h5py 폴백 경로에서:
    1. Elements_t 셀 연결성 → 점 구름이 아닌 진짜 UnstructuredGrid
    2. ZoneBC_t → OpenFOAM 리더와 동일한 계약의
       ``metadata["boundary_patches"]`` / ``metadata["boundary_patch_meshes"]``
    3. BCWall* 계열 → ``metadata["auto_wall_patches"]`` 자동 분류
    4. MIXED/NGON 등 미지원 섹션은 건너뛰고, Elements_t 가 없으면 기존
       점 구름 폴백 유지 (하위 호환)
을 검증한다. 실제 CGNS 파일 없이 pyCGNS tree fixture(순수 리스트)와
h5py 로 직접 생성한 최소 CGNS HDF5 파일만 사용한다.

실행::

    pytest tests/test_cgns_zonebc.py -q
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pyvista = pytest.importorskip("pyvista", reason="pyvista 가 필요합니다")

VTK_QUAD = 9

# ---------------------------------------------------------------------------
# 공통 fixture 데이터 — 3x3 점, QUAD_4 4개짜리 2D 격자
#
#   6 -- 7 -- 8
#   |    |    |
#   3 -- 4 -- 5
#   |    |    |
#   0 -- 1 -- 2   (0-based; CGNS 연결성은 1-based)
# ---------------------------------------------------------------------------

_X = np.array([0.0, 1.0, 2.0] * 3)
_Y = np.repeat([0.0, 1.0, 2.0], 3)
_Z = np.zeros(9)

#: QUAD_4 연결성 (1-based, CCW)
_QUAD_CONN = np.array(
    [
        1, 2, 5, 4,
        2, 3, 6, 5,
        4, 5, 8, 7,
        5, 6, 9, 8,
    ],
    dtype=np.int64,
)

#: 아래쪽 변 (wall) — 1-based 점 인덱스
_WALL_POINTS = np.array([1, 2, 3], dtype=np.int64)

#: 왼쪽 변 (inlet) — 1-based 점 인덱스
_INLET_POINTS = np.array([1, 4, 7], dtype=np.int64)

_PRESSURE = np.linspace(0.0, 8.0, 9)


def _bc_bytes(s: str) -> Any:
    """BC 타입 문자열을 pyCGNS 방식(dtype='S1' 문자 배열)으로 인코드한다."""
    return np.frombuffer(s.encode("utf-8"), dtype="S1").copy()


# ---------------------------------------------------------------------------
# pyCGNS tree fixture — [name, value, children, type] 리스트 구조
# ---------------------------------------------------------------------------


def _make_pycgns_tree(
    *,
    with_elements: bool = True,
    with_zonebc: bool = True,
    extra_sections: list[Any] | None = None,
) -> list[Any]:
    """최소 CGNS 트리(pure list)를 구성한다."""
    zone_children: list[Any] = [
        [
            "GridCoordinates",
            None,
            [
                ["CoordinateX", _X.copy(), [], "DataArray_t"],
                ["CoordinateY", _Y.copy(), [], "DataArray_t"],
                ["CoordinateZ", _Z.copy(), [], "DataArray_t"],
            ],
            "GridCoordinates_t",
        ],
        [
            "FlowSolution",
            None,
            [
                ["p", _PRESSURE.copy(), [], "DataArray_t"],
            ],
            "FlowSolution_t",
        ],
    ]

    if with_elements:
        zone_children.append(
            [
                "GridElements",
                np.array([7, 0], dtype=np.int32),  # QUAD_4=7
                [
                    ["ElementRange", np.array([1, 4], dtype=np.int64), [], "IndexRange_t"],
                    ["ElementConnectivity", _QUAD_CONN.copy(), [], "DataArray_t"],
                ],
                "Elements_t",
            ]
        )
    if extra_sections:
        zone_children.extend(extra_sections)

    if with_zonebc:
        zone_children.append(
            [
                "ZoneBC",
                None,
                [
                    [
                        "wall",
                        _bc_bytes("BCWallViscous"),
                        [
                            ["PointList", _WALL_POINTS.copy(), [], "IndexArray_t"],
                            ["GridLocation", _bc_bytes("Vertex"), [], "GridLocation_t"],
                        ],
                        "BC_t",
                    ],
                    [
                        "inlet",
                        _bc_bytes("BCInflow"),
                        [
                            ["PointList", _INLET_POINTS.copy(), [], "IndexArray_t"],
                        ],
                        "BC_t",
                    ],
                    [
                        "top",
                        _bc_bytes("BCWall"),
                        [
                            ["PointRange", np.array([[7, 9]], dtype=np.int64), [], "IndexRange_t"],
                        ],
                        "BC_t",
                    ],
                ],
                "ZoneBC_t",
            ]
        )

    zone = ["Zone", np.array([[9, 4, 0]], dtype=np.int64), zone_children, "Zone_t"]
    base = ["Base", np.array([2, 3], dtype=np.int32), [zone], "CGNSBase_t"]
    return ["CGNSTree", None, [base], "CGNSTree_t"]


# ===========================================================================
# pyCGNS tree 경로
# ===========================================================================


class TestPyCGNSTreeConnectivity:
    def test_quad_cells_built(self) -> None:
        """Elements_t QUAD_4 섹션이 진짜 셀로 변환되어야 한다 (점 구름 아님)."""
        from naviertwin.core.cfd_reader.cgns_reader import _cgns_tree_to_cfd_dataset

        dataset = _cgns_tree_to_cfd_dataset(_make_pycgns_tree())
        assert dataset.n_points == 9
        assert dataset.n_cells == 4
        assert set(np.unique(dataset.mesh.celltypes)) == {VTK_QUAD}

    def test_field_loaded_as_point_data(self) -> None:
        """FlowSolution 의 p 필드가 point_data 로 로드되어야 한다."""
        from naviertwin.core.cfd_reader.cgns_reader import _cgns_tree_to_cfd_dataset

        dataset = _cgns_tree_to_cfd_dataset(_make_pycgns_tree())
        assert "p" in dataset.field_names
        assert "p" in dataset.mesh.point_data
        np.testing.assert_allclose(dataset.mesh.point_data["p"], _PRESSURE)

    def test_wall_patch_detected(self) -> None:
        """BCWall* 계열 BC 가 is_wall=True 로 분류되어야 한다."""
        from naviertwin.core.cfd_reader.cgns_reader import _cgns_tree_to_cfd_dataset

        dataset = _cgns_tree_to_cfd_dataset(_make_pycgns_tree())
        patches = dataset.metadata["boundary_patches"]
        assert patches["wall"]["is_wall"] is True
        assert patches["wall"]["type"] == "BCWallViscous"
        assert patches["top"]["is_wall"] is True  # BCWall + PointRange
        assert patches["inlet"]["is_wall"] is False
        assert patches["inlet"]["type"] == "BCInflow"

    def test_auto_wall_patches_listed(self) -> None:
        """wall 계열 patch 이름이 auto_wall_patches 에 모여야 한다."""
        from naviertwin.core.cfd_reader.cgns_reader import _cgns_tree_to_cfd_dataset

        dataset = _cgns_tree_to_cfd_dataset(_make_pycgns_tree())
        assert dataset.metadata["auto_wall_patches"] == ["top", "wall"]

    def test_patch_mesh_contract(self) -> None:
        """boundary_patch_meshes 가 OpenFOAM 리더와 동일 계약이어야 한다."""
        from naviertwin.core.cfd_reader.cgns_reader import _cgns_tree_to_cfd_dataset

        dataset = _cgns_tree_to_cfd_dataset(_make_pycgns_tree())
        patch_meshes = dataset.metadata["boundary_patch_meshes"]
        assert "wall" in patch_meshes
        wall_mesh = patch_meshes["wall"]
        assert int(wall_mesh.n_points) == 3
        # 계약: boundary_patches 의 n_cells/n_points 는 int
        info = dataset.metadata["boundary_patches"]["wall"]
        assert isinstance(info["n_cells"], int)
        assert isinstance(info["n_points"], int)
        assert info["n_points"] == 3

    def test_point_range_bc(self) -> None:
        """PointRange 기반 BC 도 점 인덱스로 변환되어야 한다."""
        from naviertwin.core.cfd_reader.cgns_reader import _cgns_tree_to_cfd_dataset

        dataset = _cgns_tree_to_cfd_dataset(_make_pycgns_tree())
        top = dataset.metadata["boundary_patches"]["top"]
        assert top["n_points"] == 3  # 점 7..9 (1-based)

    def test_mixed_section_loaded(self) -> None:
        """MIXED(20) 안의 TRI_3 요소도 정상 변환되어야 한다."""
        from naviertwin.core.cfd_reader.cgns_reader import _cgns_tree_to_cfd_dataset

        mixed = [
            "MixedElements",
            np.array([20, 0], dtype=np.int32),  # MIXED=20
            [
                ["ElementRange", np.array([5, 6], dtype=np.int64), [], "IndexRange_t"],
                [
                    "ElementConnectivity",
                    np.array([5, 1, 2, 5, 5, 2, 3, 6], dtype=np.int64),
                    [],
                    "DataArray_t",
                ],
            ],
            "Elements_t",
        ]
        dataset = _cgns_tree_to_cfd_dataset(
            _make_pycgns_tree(extra_sections=[mixed])
        )
        assert dataset.n_cells == 6
        assert set(np.unique(dataset.mesh.celltypes)) == {5, VTK_QUAD}

    def test_point_cloud_fallback_without_elements(self) -> None:
        """Elements_t 가 없으면 기존 점 구름 폴백을 유지해야 한다 (하위 호환)."""
        from naviertwin.core.cfd_reader.cgns_reader import _cgns_tree_to_cfd_dataset

        dataset = _cgns_tree_to_cfd_dataset(
            _make_pycgns_tree(with_elements=False, with_zonebc=False)
        )
        assert dataset.n_points == 9
        assert "boundary_patches" not in dataset.metadata
        assert "p" in dataset.field_names


# ===========================================================================
# h5py 폴백 경로 — 최소 CGNS HDF5 파일을 직접 생성해 검증
# ===========================================================================


def _write_cgns_h5(path: Path) -> None:
    """h5py 로 최소 CGNS HDF5 파일을 생성한다 (3x3 점, QUAD_4 4개)."""
    h5py = pytest.importorskip("h5py", reason="h5py 가 필요합니다")

    def _node(parent: Any, name: str, label: str, data: Any = None) -> Any:
        group = parent.create_group(name)
        group.attrs["label"] = np.bytes_(label)
        if data is not None:
            group.create_dataset(" data", data=data)
        return group

    with h5py.File(str(path), "w") as f:
        base = _node(f, "Base", "CGNSBase_t", np.array([2, 3], dtype=np.int32))
        zone = _node(base, "Zone", "Zone_t", np.array([[9, 4, 0]], dtype=np.int64))

        grid = _node(zone, "GridCoordinates", "GridCoordinates_t")
        _node(grid, "CoordinateX", "DataArray_t", _X)
        _node(grid, "CoordinateY", "DataArray_t", _Y)
        _node(grid, "CoordinateZ", "DataArray_t", _Z)

        elems = _node(zone, "GridElements", "Elements_t", np.array([7, 0], dtype=np.int32))
        _node(elems, "ElementRange", "IndexRange_t", np.array([1, 4], dtype=np.int64))
        _node(elems, "ElementConnectivity", "DataArray_t", _QUAD_CONN)

        zonebc = _node(zone, "ZoneBC", "ZoneBC_t")
        wall = _node(
            zonebc,
            "wall",
            "BC_t",
            np.frombuffer(b"BCWallViscous", dtype=np.int8).copy(),
        )
        _node(wall, "PointList", "IndexArray_t", _WALL_POINTS)
        _node(
            wall,
            "GridLocation",
            "GridLocation_t",
            np.frombuffer(b"Vertex", dtype=np.int8).copy(),
        )
        inlet = _node(
            zonebc, "inlet", "BC_t", np.frombuffer(b"BCInflow", dtype=np.int8).copy()
        )
        _node(inlet, "PointList", "IndexArray_t", _INLET_POINTS)

        sol = _node(zone, "FlowSolution", "FlowSolution_t")
        _node(sol, "p", "DataArray_t", _PRESSURE)


class TestH5pyZoneBC:
    def test_h5py_direct_parse(self, tmp_path: Path) -> None:
        """h5py 폴백 파서가 연결성 + ZoneBC 를 모두 읽어야 한다."""
        h5py = pytest.importorskip("h5py", reason="h5py 가 필요합니다")
        from naviertwin.core.cfd_reader.cgns_reader import _h5py_cgns_to_cfd_dataset

        cgns_path = tmp_path / "mini_zonebc.cgns"
        _write_cgns_h5(cgns_path)

        with h5py.File(str(cgns_path), "r") as f:
            dataset = _h5py_cgns_to_cfd_dataset(f, str(cgns_path))

        # (a) 점 구름이 아니라 QUAD 셀 4개
        assert dataset.n_points == 9
        assert dataset.n_cells == 4
        assert set(np.unique(dataset.mesh.celltypes)) == {VTK_QUAD}
        # (b) wall patch 가 is_wall=True 로 존재
        patches = dataset.metadata["boundary_patches"]
        assert patches["wall"]["is_wall"] is True
        assert patches["wall"]["type"] == "BCWallViscous"
        assert patches["inlet"]["is_wall"] is False
        # (c) auto_wall_patches 에 이름 포함
        assert dataset.metadata["auto_wall_patches"] == ["wall"]
        # (d) 필드 p 로드
        assert "p" in dataset.field_names
        np.testing.assert_allclose(dataset.mesh.point_data["p"], _PRESSURE)

    def test_reader_h5py_fallback_path(self, tmp_path: Path) -> None:
        """CGNSReader.read 의 h5py 폴백 경로로 동일 결과가 나와야 한다."""
        pytest.importorskip("h5py", reason="h5py 가 필요합니다")
        from unittest.mock import patch

        from naviertwin.core.cfd_reader.cgns_reader import CGNSReader

        cgns_path = tmp_path / "mini_zonebc.cgns"
        _write_cgns_h5(cgns_path)

        reader = CGNSReader()

        def raise_always(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("mock — h5py 경로 강제")

        with patch.object(reader, "_read_with_pyvista", side_effect=raise_always):
            with patch.object(reader, "_read_with_pycgns", side_effect=raise_always):
                dataset = reader.read(cgns_path)

        assert dataset.metadata["reader"] == "h5py/CGNS"
        assert dataset.n_cells == 4
        assert dataset.metadata["boundary_patches"]["wall"]["is_wall"] is True
        assert "wall" in dataset.metadata["auto_wall_patches"]
        assert "boundary_patch_meshes" in dataset.metadata
        assert int(dataset.metadata["boundary_patch_meshes"]["wall"].n_points) == 3

    def test_h5py_backward_compat_point_cloud(self, tmp_path: Path) -> None:
        """Elements_t 없는 파일은 기존 점 구름 폴백을 유지해야 한다."""
        h5py = pytest.importorskip("h5py", reason="h5py 가 필요합니다")
        from naviertwin.core.cfd_reader.cgns_reader import _h5py_cgns_to_cfd_dataset

        cgns_path = tmp_path / "mini_pointcloud.cgns"
        with h5py.File(str(cgns_path), "w") as f:
            base = f.create_group("Base")
            base.attrs["label"] = np.bytes_("CGNSBase_t")
            zone = base.create_group("Zone")
            zone.attrs["label"] = np.bytes_("Zone_t")
            grid = zone.create_group("GridCoordinates")
            grid.attrs["label"] = np.bytes_("GridCoordinates_t")
            for cname, cdata in (
                ("CoordinateX", _X),
                ("CoordinateY", _Y),
                ("CoordinateZ", _Z),
            ):
                cgroup = grid.create_group(cname)
                cgroup.attrs["label"] = np.bytes_("DataArray_t")
                cgroup.create_dataset(" data", data=cdata)

        with h5py.File(str(cgns_path), "r") as f:
            dataset = _h5py_cgns_to_cfd_dataset(f, str(cgns_path))

        assert dataset.n_points == 9
        assert "boundary_patches" not in dataset.metadata


# ===========================================================================
# 실제 pyCGNS 로 저장한 파일 round-trip (설치 환경에서만)
# ===========================================================================


class TestRealPyCGNSRoundTrip:
    def test_pycgns_save_load_roundtrip(self, tmp_path: Path) -> None:
        """실제 CGNS.MAP 으로 저장한 파일을 pyCGNS 경로로 읽어야 한다."""
        MAP = pytest.importorskip("CGNS.MAP", reason="pyCGNS 필요")
        from unittest.mock import patch

        from naviertwin.core.cfd_reader.cgns_reader import CGNSReader

        tree = _make_pycgns_tree()
        cgns_path = tmp_path / "roundtrip.cgns"
        try:
            MAP.save(str(cgns_path), tree)
        except Exception as e:  # noqa: BLE001 — pyCGNS 버전별 저장 제약 보호
            pytest.skip(f"CGNS.MAP.save 가 이 tree 를 지원하지 않음: {e}")

        reader = CGNSReader()

        def raise_pv(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("pyvista mock — pyCGNS 경로 강제")

        with patch.object(reader, "_read_with_pyvista", side_effect=raise_pv):
            dataset = reader.read(cgns_path)

        assert dataset.n_cells == 4
        assert dataset.metadata["boundary_patches"]["wall"]["is_wall"] is True
        assert "wall" in dataset.metadata["auto_wall_patches"]
