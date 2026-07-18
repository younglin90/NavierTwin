"""Dependency policy regression tests for customer-safe install extras."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

RESTRICTED_DEPENDENCY_PREFIXES = {
    "dedalus",
    "dlrbnicsx",
    "firedrake",
    "flowtorch",
    "foamlib",
    "fluidsimfoam",
    "gmsh",
    "nlopt",
    "openturns",
    "pycgns",
    "pygmo",
    "pypdaf",
    "pymeshlab",
    "rbnicsx",
}

RESTRICTED_ALLOWED_EXTRAS = {"full"}


def _normalize_name(name: str) -> str:
    return name.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0].split("<", 1)[0].strip().lower()


def _optional_dependencies() -> dict[str, list[str]]:
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def test_restricted_dependencies_do_not_enter_core_or_dev_extras() -> None:
    """GPL/LGPL or external-solver style packages stay out of core/dev installs."""
    optional = _optional_dependencies()
    violations: list[str] = []

    for extra, deps in optional.items():
        if extra in RESTRICTED_ALLOWED_EXTRAS:
            continue
        for dep in deps:
            name = _normalize_name(dep)
            if name in RESTRICTED_DEPENDENCY_PREFIXES:
                violations.append(f"{extra}: {dep}")

    assert violations == []


def test_restricted_dependencies_are_isolated_in_full_extra() -> None:
    """The broad research stack can retain restricted packages behind full only."""
    optional = _optional_dependencies()
    full_deps = {_normalize_name(dep) for dep in optional.get("full", [])}

    expected_full_only = {
        "dedalus",
        "foamlib",
        "fluidsimfoam",
        "gmsh",
        "nlopt",
        "openturns",
        "pycgns",
        "pygmo",
        "pymeshlab",
    }

    assert expected_full_only <= full_deps


def test_license_report_script_writes_customer_due_diligence_json(tmp_path) -> None:
    """The customer due-diligence report mirrors the dependency policy checks."""
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "license-report.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/license_report.py",
            "--json",
            "--output",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_payload = json.loads(result.stdout)
    file_payload = json.loads(output.read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert stdout_payload == file_payload
    assert file_payload["status"] == "ok"
    assert file_payload["policy"] == "mit-core-with-restricted-dependencies-isolated-to-full-extra"
    assert file_payload["errors"] == []
    assert {check["name"] for check in file_payload["checks"]} == {
        "license_metadata",
        "restricted_dependency_isolation",
    }
