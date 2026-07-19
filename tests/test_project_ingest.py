"""Source manifest ingestion tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from naviertwin.core.data_model import (
    CaseSetSource,
    CaseSource,
    ProjectSource,
    ingest_project_source_manifest,
    load_project_source_manifest,
    save_project_source_manifest,
)
from naviertwin.core.storage.canonical_cache import CanonicalCache


def _write_vtu(path: Path, pressure: float) -> None:
    pv = pytest.importorskip("pyvista")
    grid = pv.ImageData(dimensions=(4, 3, 1)).cast_to_unstructured_grid()
    grid.point_data["p"] = np.full(grid.n_points, pressure, dtype=np.float32)
    grid.point_data["U"] = np.column_stack(
        [
            np.full(grid.n_points, pressure, dtype=np.float32),
            np.zeros(grid.n_points, dtype=np.float32),
            np.zeros(grid.n_points, dtype=np.float32),
        ]
    )
    grid.save(path)


def test_relative_source_manifest_builds_condition_sweep(tmp_path: Path) -> None:
    _write_vtu(tmp_path / "low.vtu", 1.0)
    _write_vtu(tmp_path / "high.vtu", 2.0)
    source = ProjectSource(
        project_id="project",
        name="condition sweep",
        case_sets=(
            CaseSetSource(
                case_set_id="sweep",
                name="inlet sweep",
                cases=(
                    CaseSource("low", "low", "low.vtu", {"inlet": 1.0}, "duct"),
                    CaseSource("high", "high", "high.vtu", {"inlet": 2.0}, "duct"),
                ),
                parameter_units={"inlet": "m/s"},
            ),
        ),
    )
    manifest_path = save_project_source_manifest(source, tmp_path / "sources.json")

    restored_source = load_project_source_manifest(manifest_path)
    project = ingest_project_source_manifest(manifest_path)

    assert restored_source == source
    assert len(project.case_sets) == 1
    assert [case.parameters["inlet"] for case in project.case_sets[0].cases] == [1.0, 2.0]
    assert len(project.meshes) == 1
    assert len(project.geometries) == 1
    assert project.meshes[0].geometry_id == "duct"
    assert project.case_sets[0].fields[1].n_components in (1, 3)


def test_source_manifest_rejects_mixed_parameter_axes() -> None:
    with pytest.raises(ValueError, match="same parameter axes"):
        CaseSetSource(
            case_set_id="bad",
            name="bad",
            cases=(
                CaseSource("a", "a", "a.vtu", {"Re": 1.0}),
                CaseSource("b", "b", "b.vtu", {"Mach": 0.1}),
            ),
        )


def test_directory_cache_key_tracks_nested_file_changes(tmp_path: Path) -> None:
    case_dir = tmp_path / "foam-case"
    field_dir = case_dir / "1"
    field_dir.mkdir(parents=True)
    field = field_dir / "U"
    field.write_text("first", encoding="utf-8")

    first = CanonicalCache.key_for(case_dir)
    stat = field.stat()
    field.write_text("second", encoding="utf-8")
    os.utime(field, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    second = CanonicalCache.key_for(case_dir)

    assert first != second


def test_empty_openfoam_marker_does_not_change_directory_key(tmp_path: Path) -> None:
    case_dir = tmp_path / "foam-case"
    (case_dir / "constant").mkdir(parents=True)
    source = case_dir / "constant" / "transportProperties"
    source.write_text("nu 1e-5;", encoding="utf-8")
    before = CanonicalCache.key_for(case_dir)

    (case_dir / "case.foam").touch()

    assert CanonicalCache.key_for(case_dir) == before
