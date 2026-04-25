"""Schema-validated metadata writer (JSON).

Examples:
    >>> from naviertwin.utils.dataset_metadata import write_metadata
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from naviertwin.utils.schema_validator import validate

_REQUIRED = {
    "name": str,
    "version": str,
    "n_samples": int,
}


def write_metadata(path: str | Path, meta: dict[str, Any]) -> None:
    ok, errs = validate(meta, _REQUIRED)
    if not ok:
        raise ValueError(f"metadata invalid: {errs}")
    Path(path).write_text(json.dumps(meta, indent=2, sort_keys=True))


def read_metadata(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


__all__ = ["read_metadata", "write_metadata"]
