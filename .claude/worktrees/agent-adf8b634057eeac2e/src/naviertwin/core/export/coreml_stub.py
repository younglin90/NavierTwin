"""CoreML stub — name-only output marker (no actual conversion).

Examples:
    >>> from naviertwin.core.export.coreml_stub import suggest_mlmodel_name
    >>> suggest_mlmodel_name('mymodel')
    'mymodel.mlmodel'
"""

from __future__ import annotations

from pathlib import Path


def suggest_mlmodel_name(name: str) -> str:
    return f"{name}.mlmodel"


def write_stub_marker(path: str | Path, *, model_name: str) -> None:
    Path(path).write_text(
        f"# CoreML stub\nmodel: {model_name}\nformat: mlmodel\n",
    )


__all__ = ["suggest_mlmodel_name", "write_stub_marker"]
