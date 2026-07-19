"""Runtime canonical workspace tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from naviertwin.core.data_model import LineageKind, TwinWorkspace, load_project_manifest


def _dataset(value: float = 1.0):
    pv = pytest.importorskip("pyvista")
    from naviertwin.core.cfd_reader.base import CFDDataset

    mesh = pv.ImageData(dimensions=(4, 3, 1)).cast_to_unstructured_grid()
    mesh.point_data["p"] = np.full(mesh.n_points, value)
    return CFDDataset(mesh=mesh, time_steps=[0.0], field_names=["p"])


def test_single_dataset_builds_canonical_project() -> None:
    workspace = TwinWorkspace()
    dataset = _dataset()

    project = workspace.load_single_dataset(dataset, name="demo", source="memory://demo")
    status = workspace.status()

    assert workspace.view_dataset is dataset
    assert workspace.case_datasets is None
    assert project.case_sets[0].cases[0].snapshots
    assert status.project_id == project.project_id
    assert status.runtime_complete


def test_case_set_owns_parameters_and_active_case() -> None:
    workspace = TwinWorkspace()
    datasets = [_dataset(1.0), _dataset(2.0)]

    workspace.load_case_set(
        datasets,
        [[10.0], [20.0]],
        ["inlet_velocity"],
        case_names=["low", "high"],
        geometry_ids=["duct", "duct"],
    )

    assert workspace.case_datasets == tuple(datasets)
    np.testing.assert_allclose(workspace.parameters, [[10.0], [20.0]])
    assert workspace.parameter_names == ("inlet_velocity",)
    assert workspace.select_case(1) is datasets[1]
    assert workspace.status().active_case_index == 1


def test_case_set_rejects_parameter_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="parameter matrix shape"):
        TwinWorkspace().load_case_set(
            [_dataset(), _dataset()],
            [[1.0]],
            ["Re"],
        )


def test_manifest_persistence_uses_workspace_project(tmp_path: Path) -> None:
    workspace = TwinWorkspace()
    workspace.load_single_dataset(_dataset(), name="persisted")

    path = workspace.save_manifest(tmp_path / "project.manifest.json")

    assert load_project_manifest(path) == workspace.project


def test_workspace_appends_durable_lineage() -> None:
    workspace = TwinWorkspace()
    workspace.load_single_dataset(_dataset(), name="tracked")

    record = workspace.record_lineage(
        LineageKind.MODEL,
        artifact_id="model-a",
        input_ids=("primary",),
        strategy="rom",
        metrics={"rmse": 0.02},
    )

    assert record.artifact_id == "model-a"
    assert workspace.project is not None
    assert workspace.project.lineage == (record,)
    assert workspace.status().revision == 2


def test_ntwin_embeds_project_lineage(tmp_path: Path) -> None:
    from naviertwin.core.export.ntwin_format import (
        load_embedded_project_manifest,
        save_dataset,
    )

    workspace = TwinWorkspace()
    dataset = _dataset()
    workspace.load_single_dataset(dataset, name="embedded")
    workspace.record_lineage(
        "prediction",
        artifact_id="prediction-a",
        parameters={"Re": 1000.0},
    )
    path = tmp_path / "embedded.ntwin"

    save_dataset(dataset, path, canonical_project=workspace.project)
    restored = load_embedded_project_manifest(path)

    assert restored == workspace.project
    assert restored.lineage[0].kind is LineageKind.PREDICTION
