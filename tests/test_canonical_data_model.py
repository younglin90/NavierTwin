"""Canonical case/geometry/time/boundary contract tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from naviertwin.core.data_model import (
    BoundaryCondition,
    BoundaryKind,
    BoundaryPatch,
    Case,
    CaseSet,
    FieldDataRef,
    FieldDescriptor,
    FieldLocation,
    GeometryDescriptor,
    LineageKind,
    LineageRecord,
    MeshDescriptor,
    Snapshot,
    TwinProject,
    load_project_manifest,
    save_project_manifest,
)


def _project() -> TwinProject:
    field = FieldDescriptor(
        name="U",
        location=FieldLocation.POINT,
        n_components=3,
        component_names=("x", "y", "z"),
        unit="m/s",
        dtype="float32",
    )
    ref = FieldDataRef(
        field_name="U",
        storage_uri="case.ntwin",
        array_path="fields/point/U",
        shape=(2, 4, 3),
        time_index=0,
        valid_mask_path="masks/point/U",
    )
    snapshot = Snapshot(
        snapshot_id="case-a:t000000",
        mesh_id="mesh-a",
        time=0.0,
        fields=(ref,),
    )
    condition = BoundaryCondition(
        patch_id="mesh-a:patch:inlet",
        kind=BoundaryKind.INLET,
        values={"U": [10.0, 0.0, 0.0]},
        units={"U": "m/s"},
    )
    case = Case(
        case_id="case-a",
        name="Re1000",
        parameters={"Re": 1000.0},
        snapshots=(snapshot,),
        boundary_conditions=(condition,),
    )
    case_set = CaseSet(
        case_set_id="sweep-a",
        name="steady sweep",
        fields=(field,),
        cases=(case,),
        parameter_units={"Re": "1"},
    )
    mesh = MeshDescriptor(
        mesh_id="mesh-a",
        geometry_id="geometry-a",
        topology_hash="topology",
        coordinate_hash="coordinates",
        topological_dim=2,
        embedding_dim=3,
        n_points=4,
        n_cells=1,
    )
    geometry = GeometryDescriptor(
        geometry_id="geometry-a",
        shape_signature="shape",
        sdf_field="sdf",
    )
    patch = BoundaryPatch(
        patch_id="mesh-a:patch:inlet",
        mesh_id="mesh-a",
        name="inlet",
        kind=BoundaryKind.INLET,
        face_ids=(0,),
    )
    return TwinProject(
        project_id="project-a",
        name="canonical test",
        case_sets=(case_set,),
        meshes=(mesh,),
        geometries=(geometry,),
        boundaries=(patch,),
    )


def test_manifest_round_trip(tmp_path: Path) -> None:
    base = _project()
    project = TwinProject(
        project_id=base.project_id,
        name=base.name,
        case_sets=base.case_sets,
        meshes=base.meshes,
        geometries=base.geometries,
        boundaries=base.boundaries,
        lineage=(
            LineageRecord(
                artifact_id="model-1",
                kind=LineageKind.MODEL,
                input_ids=("sweep-a",),
                strategy="rom",
                metrics={"rmse": 0.01},
            ),
        ),
    )
    path = save_project_manifest(project, tmp_path / "project.json")

    restored = load_project_manifest(path)

    assert restored == project
    assert restored.case_sets[0].fields[0].n_components == 3
    assert restored.case_sets[0].cases[0].snapshots[0].fields[0].valid_mask_path
    assert restored.lineage[0].kind is LineageKind.MODEL


def test_project_rejects_duplicate_lineage_ids() -> None:
    project = _project()
    record = LineageRecord("same", LineageKind.MAPPING)

    with pytest.raises(ValueError, match="duplicate lineage artifact"):
        TwinProject(
            project_id=project.project_id,
            name=project.name,
            case_sets=project.case_sets,
            meshes=project.meshes,
            geometries=project.geometries,
            lineage=(record, record),
        )


def test_project_rejects_unknown_mesh_reference() -> None:
    project = _project()
    bad_snapshot = Snapshot(
        snapshot_id="bad",
        mesh_id="missing",
        fields=project.case_sets[0].cases[0].snapshots[0].fields,
    )
    bad_case = Case(
        case_id="bad",
        name="bad",
        parameters={"Re": 1000.0},
        snapshots=(bad_snapshot,),
    )
    bad_set = CaseSet(
        case_set_id="bad",
        name="bad",
        fields=project.case_sets[0].fields,
        cases=(bad_case,),
        parameter_units={"Re": "1"},
    )

    with pytest.raises(ValueError, match="unknown mesh"):
        TwinProject(
            project_id="bad",
            name="bad",
            case_sets=(bad_set,),
            meshes=project.meshes,
            geometries=project.geometries,
        )


def test_case_set_rejects_inconsistent_parameter_axes() -> None:
    snapshot = _project().case_sets[0].cases[0].snapshots[0]
    first = Case("a", "a", {"Re": 1.0}, (snapshot,))
    second = Case("b", "b", {"Mach": 0.2}, (snapshot,))

    with pytest.raises(ValueError, match="same parameters"):
        CaseSet("set", "set", (), (first, second))


def test_case_rejects_unordered_time_axis() -> None:
    ref = _project().case_sets[0].cases[0].snapshots[0].fields
    snapshots = (
        Snapshot("late", "mesh-a", ref, time=1.0),
        Snapshot("early", "mesh-a", ref, time=0.0),
    )
    with pytest.raises(ValueError, match="increasing"):
        Case("case", "case", {}, snapshots)


def test_legacy_adapter_preserves_vector_contract_and_time() -> None:
    pv = pytest.importorskip("pyvista")
    from naviertwin.core.cfd_reader.base import CFDDataset
    from naviertwin.core.data_model import case_set_from_cfd_datasets

    grid = pv.ImageData(dimensions=(3, 3, 1)).cast_to_unstructured_grid()
    n_points = grid.n_points
    dataset = CFDDataset(
        mesh=grid,
        time_steps=[0.0, 0.5],
        field_names=["U", "p"],
        metadata={
            "time_series_fields": {
                "U": np.zeros((2, n_points, 3), dtype=np.float32),
                "p": np.zeros((2, n_points), dtype=np.float64),
            },
            "time_series_locations": {"U": "point", "p": "point"},
            "field_units": {"U": "m/s", "p": "Pa"},
            "boundary_patches": {
                "inlet": {"type": "inlet", "n_cells": 2, "n_points": 3}
            },
            "boundary_conditions": {
                "inlet": {"type": "inlet", "values": {"U": [1.0, 0.0, 0.0]}}
            },
        },
    )

    case_set, meshes, geometries, boundaries = case_set_from_cfd_datasets(
        [dataset],
        case_set_id="set-a",
        name="unsteady",
        parameters=[{"Re": 1000.0}],
    )

    descriptors = {item.name: item for item in case_set.fields}
    assert descriptors["U"].n_components == 3
    assert descriptors["U"].component_names == ("x", "y", "z")
    assert descriptors["p"].n_components == 1
    assert [item.time for item in case_set.cases[0].snapshots] == [0.0, 0.5]
    assert len(meshes) == len(geometries) == len(boundaries) == 1
    assert case_set.cases[0].boundary_conditions[0].kind is BoundaryKind.INLET
    assert case_set.cases[0].snapshots[0].fields[0].valid_mask_path is None


def test_legacy_adapter_deduplicates_mesh_across_result_sources() -> None:
    pv = pytest.importorskip("pyvista")
    from naviertwin.core.cfd_reader.base import CFDDataset
    from naviertwin.core.data_model import case_set_from_cfd_datasets

    mesh = pv.ImageData(dimensions=(3, 3, 1)).cast_to_unstructured_grid()
    mesh.point_data["p"] = np.ones(mesh.n_points)
    first = CFDDataset(
        mesh=mesh.copy(),
        time_steps=np.array([0.0]),
        field_names=["p"],
        metadata={"source_file": "first.vtu"},
    )
    second = CFDDataset(
        mesh=mesh.copy(),
        time_steps=np.array([0.0]),
        field_names=["p"],
        metadata={"source_file": "second.vtu"},
    )

    _, meshes, geometries, _ = case_set_from_cfd_datasets(
        [first, second],
        case_set_id="shared-mesh",
        name="shared mesh",
        geometry_ids=["duct", "duct"],
    )

    assert len(meshes) == 1
    assert len(geometries) == 1
    assert meshes[0].source_uri == ""
