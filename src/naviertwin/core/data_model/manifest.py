"""Atomic JSON persistence for canonical project manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from naviertwin.core.data_model.schema import TwinProject, project_from_dict
from naviertwin.utils.atomic_io import atomic_write_text
from naviertwin.utils.json_safe import safe_dumps


def save_project_manifest(project: TwinProject, path: str | Path) -> Path:
    """Validate and atomically save a project manifest."""

    # Construction validates the full graph. Round-tripping through the parser
    # also catches values that cannot satisfy the persisted schema.
    payload = project.to_dict()
    project_from_dict(payload)
    return atomic_write_text(path, safe_dumps(payload, sort_keys=True) + "\n")


def load_project_manifest(path: str | Path) -> TwinProject:
    """Load and validate a project manifest."""

    source = Path(path)
    payload: Any = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("project manifest root must be an object")
    return project_from_dict(payload)


__all__ = ["load_project_manifest", "save_project_manifest"]
