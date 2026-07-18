"""Helpers for release/update-check smoke version inputs."""

from __future__ import annotations

import json
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_version() -> str:
    """Return the current project version from pyproject.toml."""
    pyproject = _repo_root() / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version")
    if not isinstance(version, str) or not version:
        raise RuntimeError("project.version is missing in pyproject.toml")
    return version


def example_latest_release_version() -> str:
    """Return the example latest release version from example metadata."""
    metadata = _repo_root() / "examples" / "release-metadata.example.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    version = payload.get("version")
    if not isinstance(version, str) or not version:
        raise RuntimeError("examples/release-metadata.example.json missing 'version'")
    return version
