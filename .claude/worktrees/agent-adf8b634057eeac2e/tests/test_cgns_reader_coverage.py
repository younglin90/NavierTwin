"""Round 599 — CGNSReader fallback chain + helper coverage."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest


def _block(*names):
    real_import = builtins.__import__

    def block(name, *a, **kw):
        if name in names or any(name.startswith(n) for n in names):
            raise ImportError(f"blocked {name}")
        return real_import(name, *a, **kw)

    return block


class TestCGNSReaderFallback:
    def test_file_not_found(self, tmp_path: Path) -> None:
        from naviertwin.core.cfd_reader.cgns_reader import CGNSReader

        reader = CGNSReader()
        with pytest.raises(FileNotFoundError):
            reader.read(tmp_path / "missing.cgns")

    def test_all_parsers_fail_raises_value_error(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from naviertwin.core.cfd_reader.cgns_reader import CGNSReader

        f = tmp_path / "fake.cgns"
        f.write_bytes(b"not a real cgns file")
        reader = CGNSReader()

        monkeypatch.setattr(builtins, "__import__", _block("pyvista", "CGNS", "h5py", "meshio"))
        with pytest.raises((ValueError, ImportError)):
            reader.read(f)

    def test_read_with_pyvista_import_error(self, tmp_path: Path, monkeypatch) -> None:
        from naviertwin.core.cfd_reader.cgns_reader import CGNSReader

        f = tmp_path / "x.cgns"
        f.touch()
        reader = CGNSReader()
        monkeypatch.setattr(builtins, "__import__", _block("pyvista"))
        with pytest.raises(ImportError, match="pyvista"):
            reader._read_with_pyvista(f)

    def test_read_with_pycgns_import_error(self, tmp_path: Path, monkeypatch) -> None:
        from naviertwin.core.cfd_reader.cgns_reader import CGNSReader

        f = tmp_path / "x.cgns"
        f.touch()
        reader = CGNSReader()
        monkeypatch.setattr(builtins, "__import__", _block("CGNS"))
        with pytest.raises(ImportError, match="pyCGNS"):
            reader._read_with_pycgns(f)

    def test_read_with_h5py_import_error(self, tmp_path: Path, monkeypatch) -> None:
        from naviertwin.core.cfd_reader.cgns_reader import CGNSReader

        f = tmp_path / "x.cgns"
        f.touch()
        reader = CGNSReader()
        monkeypatch.setattr(builtins, "__import__", _block("h5py"))
        with pytest.raises(ImportError, match="h5py"):
            reader._read_with_h5py(f)

    def test_read_with_meshio_import_error(self, tmp_path: Path, monkeypatch) -> None:
        from naviertwin.core.cfd_reader.cgns_reader import CGNSReader

        f = tmp_path / "x.cgns"
        f.touch()
        reader = CGNSReader()
        monkeypatch.setattr(builtins, "__import__", _block("meshio"))
        with pytest.raises(ImportError, match="meshio"):
            reader._read_with_meshio(f)


class TestCGNSTreeHelpers:
    def test_cgns_tree_to_dataset_no_coords_raises(self) -> None:
        from naviertwin.core.cfd_reader.cgns_reader import _cgns_tree_to_cfd_dataset

        # empty tree has no GridCoordinates
        empty_tree = ["root", None, [], "CGNSBase_t"]
        with pytest.raises((ValueError, ImportError)):
            _cgns_tree_to_cfd_dataset(empty_tree)

    def test_cgns_tree_grid_coordinates(self) -> None:
        pytest.importorskip("pyvista")
        import numpy as np

        from naviertwin.core.cfd_reader.cgns_reader import _cgns_tree_to_cfd_dataset

        coords_x = np.array([0.0, 1.0, 2.0])
        coords_y = np.array([0.0, 0.0, 0.0])
        gc_children = [
            ["CoordinateX", coords_x, [], "DataArray_t"],
            ["CoordinateY", coords_y, [], "DataArray_t"],
        ]
        flow_children = [
            ["p", np.array([1.0, 2.0, 3.0]), [], "DataArray_t"],
        ]
        zone = ["Zone1", None, [
            ["GridCoordinates", None, gc_children, "GridCoordinates_t"],
            ["FlowSolution", None, flow_children, "FlowSolution_t"],
        ], "Zone_t"]
        base = ["Base", None, [zone], "CGNSBase_t"]
        tree = ["tree", None, [base], "CGNSTree_t"]

        ds = _cgns_tree_to_cfd_dataset(tree)
        assert "p" in ds.field_names
        assert ds.mesh.n_points == 3
