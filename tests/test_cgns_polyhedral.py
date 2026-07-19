"""CGNS MIXED/NGON/NFACE conversion tests."""

from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from naviertwin.core.cfd_reader.cgns_reader import _sections_to_vtk_cells  # noqa: E402


def test_mixed_section_converts_each_embedded_type() -> None:
    sections = [
        {
            "name": "mixed",
            "etype": 20,
            "conn": np.array([5, 1, 2, 3, 7, 1, 2, 4, 3]),
            "one_based": True,
        }
    ]
    cells, types = _sections_to_vtk_cells(sections, 4)
    mesh = pv.UnstructuredGrid(cells, types, np.zeros((4, 3)))
    assert mesh.n_cells == 2
    assert types.tolist() == [5, 9]


def test_ngon_without_nface_becomes_polygon_cells() -> None:
    sections = [
        {
            "name": "faces",
            "etype": 22,
            "conn": np.array([1, 2, 3, 1, 3, 4]),
            "offsets": np.array([0, 3, 6]),
            "range": np.array([1, 2]),
            "one_based": True,
        }
    ]
    cells, types = _sections_to_vtk_cells(sections, 4)
    mesh = pv.UnstructuredGrid(cells, types, np.zeros((4, 3)))
    assert mesh.n_cells == 2
    assert types.tolist() == [7, 7]


def test_ngon_nface_builds_vtk_polyhedron() -> None:
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            1, 4, 3, 2,
            5, 6, 7, 8,
            1, 2, 6, 5,
            2, 3, 7, 6,
            3, 4, 8, 7,
            4, 1, 5, 8,
        ]
    )
    sections = [
        {
            "name": "faces",
            "zone": "zone",
            "etype": 22,
            "conn": faces,
            "offsets": np.arange(0, 25, 4),
            "range": np.array([1, 6]),
            "one_based": True,
        },
        {
            "name": "cells",
            "zone": "zone",
            "etype": 23,
            "conn": np.arange(1, 7),
            "offsets": np.array([0, 6]),
            "range": np.array([7, 7]),
            "one_based": True,
        },
    ]
    cells, types = _sections_to_vtk_cells(sections, len(points))
    mesh = pv.UnstructuredGrid(cells, types, points)
    assert mesh.n_cells == 1
    assert types.tolist() == [42]
    assert mesh.volume == pytest.approx(1.0)


def _zone_tree(name: str, x_offset: float) -> list:
    points_x = np.array([0.0, 1.0, 1.0, 0.0]) + x_offset
    points_y = np.array([0.0, 0.0, 1.0, 1.0])
    grid = [
        "GridCoordinates",
        None,
        [
            ["CoordinateX", points_x, [], "DataArray_t"],
            ["CoordinateY", points_y, [], "DataArray_t"],
        ],
        "GridCoordinates_t",
    ]
    elements = [
        "Elements",
        np.array([7, 0]),
        [
            ["ElementRange", np.array([1, 1]), [], "IndexRange_t"],
            ["ElementConnectivity", np.array([1, 2, 3, 4]), [], "DataArray_t"],
        ],
        "Elements_t",
    ]
    solution = [
        "FlowSolution",
        None,
        [["p", np.full(4, x_offset), [], "DataArray_t"]],
        "FlowSolution_t",
    ]
    wall = [
        "ZoneBC",
        None,
        [
            [
                "wall",
                np.frombuffer(b"BCWall", dtype="S1"),
                [["PointList", np.array([1, 2]), [], "IndexArray_t"]],
                "BC_t",
            ]
        ],
        "ZoneBC_t",
    ]
    return [name, np.array([[4, 1, 0]]), [grid, elements, solution, wall], "Zone_t"]


def test_pycgns_multizone_preserves_connectivity_fields_and_bc_namespace() -> None:
    from naviertwin.core.cfd_reader.cgns_reader import _cgns_tree_to_cfd_dataset

    tree = [
        "CGNSTree",
        None,
        [["Base", np.array([2, 3]), [_zone_tree("A", 0.0), _zone_tree("B", 2.0)], "CGNSBase_t"]],
        "CGNSTree_t",
    ]
    dataset = _cgns_tree_to_cfd_dataset(tree)
    assert dataset.n_points == 8
    assert dataset.n_cells == 2
    np.testing.assert_allclose(dataset.mesh.point_data["p"], [0] * 4 + [2] * 4)
    assert dataset.metadata["zone_names"] == ["A", "B"]
    assert set(dataset.metadata["boundary_patches"]) == {"A/wall", "B/wall"}


def test_h5py_multizone_preserves_connectivity_fields_and_bc_namespace(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    from naviertwin.core.cfd_reader.cgns_reader import _h5py_cgns_to_cfd_dataset

    path = tmp_path / "multizone.cgns"

    def node(parent, name, label, data=None):
        group = parent.create_group(name)
        group.attrs["label"] = np.bytes_(label)
        if data is not None:
            group.create_dataset(" data", data=data)
        return group

    with h5py.File(path, "w") as handle:
        base = node(handle, "Base", "CGNSBase_t", np.array([2, 3]))
        for zone_name, x_offset in (("A", 0.0), ("B", 2.0)):
            zone = node(base, zone_name, "Zone_t", np.array([[4, 1, 0]]))
            grid = node(zone, "GridCoordinates", "GridCoordinates_t")
            node(
                grid,
                "CoordinateX",
                "DataArray_t",
                np.array([0.0, 1.0, 1.0, 0.0]) + x_offset,
            )
            node(grid, "CoordinateY", "DataArray_t", np.array([0.0, 0.0, 1.0, 1.0]))
            elements = node(zone, "Elements", "Elements_t", np.array([7, 0]))
            node(elements, "ElementRange", "IndexRange_t", np.array([1, 1]))
            node(
                elements,
                "ElementConnectivity",
                "DataArray_t",
                np.array([1, 2, 3, 4]),
            )
            solution = node(zone, "FlowSolution", "FlowSolution_t")
            node(solution, "p", "DataArray_t", np.full(4, x_offset))
            zone_bc = node(zone, "ZoneBC", "ZoneBC_t")
            wall = node(
                zone_bc,
                "wall",
                "BC_t",
                np.frombuffer(b"BCWall", dtype=np.int8),
            )
            node(wall, "PointList", "IndexArray_t", np.array([1, 2]))

    with h5py.File(path, "r") as handle:
        dataset = _h5py_cgns_to_cfd_dataset(handle, str(path))
    assert dataset.n_points == 8
    assert dataset.n_cells == 2
    np.testing.assert_allclose(dataset.mesh.point_data["p"], [0] * 4 + [2] * 4)
    assert dataset.metadata["zone_count"] == 2
    assert set(dataset.metadata["auto_wall_patches"]) == {"A/wall", "B/wall"}
