"""TFLite stub — name + marker.

Examples:
    >>> from naviertwin.core.export.tflite_stub import suggest_tflite_name
    >>> suggest_tflite_name('m')
    'm.tflite'
"""

from __future__ import annotations

from pathlib import Path


def suggest_tflite_name(name: str) -> str:
    return f"{name}.tflite"


def write_stub_marker(path: str | Path, *, model_name: str) -> None:
    Path(path).write_text(
        f"# TFLite stub\nmodel: {model_name}\nformat: tflite\n",
    )


__all__ = ["suggest_tflite_name", "write_stub_marker"]
