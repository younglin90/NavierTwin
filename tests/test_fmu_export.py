"""FMI/FMU export adapter tests."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest


def _fitted_engine() -> object:
    pytest.importorskip("sklearn")
    from naviertwin.core.digital_twin.twin_engine import TwinEngine

    rng = np.random.default_rng(11)
    params = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
    basis = rng.standard_normal((12, 2))
    coeffs = np.vstack([
        np.sin(np.pi * params[:, 0]),
        np.cos(np.pi * params[:, 0]),
    ])
    snapshots = basis @ coeffs

    engine = TwinEngine(reducer_type="pod", surrogate_type="rbf", n_modes=2)
    engine.fit(snapshots, params)
    return engine


def test_export_to_fmu_writes_fmi_archive(tmp_path: Path) -> None:
    from naviertwin.core.export.fmu_export import (
        export_to_fmu,
        inspect_fmu,
        validate_fmu_archive,
    )

    out = tmp_path / "naviertwin.fmu"
    info = export_to_fmu(
        _fitted_engine(),
        out,
        model_name="NavierTwinROM",
        input_names=["mach"],
        output_names=["pressure", "velocity"],
    )

    assert info.path == out
    assert out.exists()
    with zipfile.ZipFile(out) as archive:
        names = set(archive.namelist())
        model_description = archive.read("modelDescription.xml").decode("utf-8")

    assert {
        "modelDescription.xml",
        "resources/naviertwin_fmu.json",
        "resources/README.txt",
        "resources/engine.pkl",
        "documentation/index.html",
    } <= names
    assert 'fmiVersion="2.0"' in model_description
    assert "<CoSimulation" in model_description
    assert 'name="mach"' in model_description
    assert 'name="pressure"' in model_description

    manifest = inspect_fmu(out)
    assert manifest["format"] == "FMI 2.0 Co-Simulation FMU"
    assert manifest["model_name"] == "NavierTwinROM"
    assert manifest["input_names"] == ["mach"]
    assert manifest["output_names"] == ["pressure", "velocity"]
    assert validate_fmu_archive(out)["status"] == "ok"


def test_validate_fmu_archive_reports_missing_entries(tmp_path: Path) -> None:
    from naviertwin.core.export.fmu_export import validate_fmu_archive

    out = tmp_path / "bad.fmu"
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("modelDescription.xml", "<fmiModelDescription/>")

    report = validate_fmu_archive(out)

    assert report["status"] == "error"
    assert "missing:" in report["errors"][0]


def test_export_to_fmu_requires_predict() -> None:
    from naviertwin.core.export.fmu_export import export_to_fmu

    with pytest.raises(RuntimeError, match="predict"):
        export_to_fmu(object(), "bad.fmu")
