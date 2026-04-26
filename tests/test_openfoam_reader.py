"""Round 594 — OpenFOAMReader helper coverage."""

from __future__ import annotations

from pathlib import Path

import pytest


def _make_case_dir(tmp_path: Path) -> Path:
    case = tmp_path / "cavity"
    case.mkdir()
    # minimal OpenFOAM structure
    (case / "constant").mkdir()
    (case / "system").mkdir()
    t0 = case / "0"
    t0.mkdir()
    (t0 / "U").write_text("// U field", encoding="utf-8")
    (t0 / "p").write_text("// p field", encoding="utf-8")
    t1 = case / "1"
    t1.mkdir()
    (t1 / "U").write_text("// U field 1s", encoding="utf-8")
    return case


class TestOpenFOAMReaderHelpers:
    def test_find_foam_file_creates_dummy(self, tmp_path: Path) -> None:
        from naviertwin.core.cfd_reader.openfoam_reader import OpenFOAMReader

        case = _make_case_dir(tmp_path)
        reader = OpenFOAMReader()
        foam = reader._find_foam_file(case)
        assert foam.suffix == ".foam"
        assert foam.exists()

    def test_find_foam_file_reuses_existing(self, tmp_path: Path) -> None:
        from naviertwin.core.cfd_reader.openfoam_reader import OpenFOAMReader

        case = _make_case_dir(tmp_path)
        existing = case / "cavity.foam"
        existing.touch()
        reader = OpenFOAMReader()
        foam = reader._find_foam_file(case)
        assert foam == existing

    def test_find_foam_file_direct_path(self, tmp_path: Path) -> None:
        from naviertwin.core.cfd_reader.openfoam_reader import OpenFOAMReader

        foam_file = tmp_path / "case.foam"
        foam_file.touch()
        reader = OpenFOAMReader()
        result = reader._find_foam_file(foam_file)
        assert result == foam_file

    def test_detect_time_steps(self, tmp_path: Path) -> None:
        from naviertwin.core.cfd_reader.openfoam_reader import OpenFOAMReader

        case = _make_case_dir(tmp_path)
        reader = OpenFOAMReader()
        ts = reader._detect_time_steps(case)
        assert sorted(ts) == [0.0, 1.0]

    def test_detect_time_steps_skips_constant_system(self, tmp_path: Path) -> None:
        from naviertwin.core.cfd_reader.openfoam_reader import OpenFOAMReader

        case = _make_case_dir(tmp_path)
        # processor dir should be skipped
        (case / "processor0").mkdir()
        reader = OpenFOAMReader()
        ts = reader._detect_time_steps(case)
        assert 0.0 in ts
        assert 1.0 in ts
        # no extra time steps from processor0
        assert len(ts) == 2

    def test_detect_field_names(self, tmp_path: Path) -> None:
        from naviertwin.core.cfd_reader.openfoam_reader import OpenFOAMReader

        case = _make_case_dir(tmp_path)
        reader = OpenFOAMReader()
        fields = reader._detect_field_names(case, "0")
        assert "U" in fields
        assert "p" in fields

    def test_detect_field_names_missing_dir(self, tmp_path: Path) -> None:
        from naviertwin.core.cfd_reader.openfoam_reader import OpenFOAMReader

        case = _make_case_dir(tmp_path)
        reader = OpenFOAMReader()
        fields = reader._detect_field_names(case, "999")
        assert fields == []

    def test_read_file_not_found(self, tmp_path: Path) -> None:
        from naviertwin.core.cfd_reader.openfoam_reader import OpenFOAMReader

        reader = OpenFOAMReader()
        with pytest.raises(FileNotFoundError):
            reader.read(tmp_path / "nonexistent_case")

    def test_read_no_pyvista_no_ofpp(self, tmp_path: Path, monkeypatch) -> None:
        import builtins

        from naviertwin.core.cfd_reader.openfoam_reader import OpenFOAMReader

        case = _make_case_dir(tmp_path)
        reader = OpenFOAMReader()

        real_import = builtins.__import__

        def block(name, *a, **kw):
            if name in ("pyvista", "ofpp"):
                raise ImportError(f"blocked {name}")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block)
        with pytest.raises(ImportError, match="pyvista|ofpp"):
            reader.read(case)


class TestModuleLevelHelpers:
    def test_collect_field_names_from_mesh(self) -> None:
        pv = pytest.importorskip("pyvista")
        import numpy as np

        from naviertwin.core.cfd_reader.openfoam_reader import _collect_field_names

        mesh = pv.Sphere()
        mesh.point_data["U"] = np.zeros((mesh.n_points, 3))
        mesh.cell_data["p"] = np.zeros(mesh.n_cells)
        names = _collect_field_names(mesh)
        assert "U" in names
        assert "p" in names

    def test_extract_unstructured_grid_from_unstructured(self) -> None:
        pv = pytest.importorskip("pyvista")
        from naviertwin.core.cfd_reader.openfoam_reader import _extract_unstructured_grid

        mesh = pv.Sphere().cast_to_unstructured_grid()
        result = _extract_unstructured_grid(mesh)
        assert isinstance(result, pv.UnstructuredGrid)

    def test_extract_unstructured_grid_none_without_pyvista(
        self, monkeypatch
    ) -> None:
        import builtins

        from naviertwin.core.cfd_reader import openfoam_reader

        real_import = builtins.__import__

        def block(name, *a, **kw):
            if name == "pyvista":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block)
        result = openfoam_reader._extract_unstructured_grid(object())
        assert result is None
