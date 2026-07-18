"""ONNX export wrapper — runtime import policy.

Examples:
    >>> from naviertwin.core.export.onnx_wrap import has_onnx
    >>> isinstance(has_onnx(), bool)
    True
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def has_onnx() -> bool:
    try:
        import onnx  # noqa: F401
    except ImportError:
        return False
    return True


def safe_export(
    model: Any, dummy_input: Any, path: str | Path,
    *, opset: int = 17,
) -> bool:
    """Returns True if exported, False if torch/onnx unavailable."""
    if not has_torch():
        return False
    import torch
    try:
        torch.onnx.export(model, dummy_input, str(path), opset_version=opset)
    except Exception:
        return False
    return True


__all__ = ["has_onnx", "has_torch", "safe_export"]
