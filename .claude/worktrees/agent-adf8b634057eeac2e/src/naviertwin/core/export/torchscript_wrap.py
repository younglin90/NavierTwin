"""TorchScript export wrapper — trace + script.

Examples:
    >>> import pytest
    >>> torch = pytest.importorskip('torch')
    >>> from naviertwin.core.export.torchscript_wrap import trace_module
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


def trace_module(model: Any, example: Any, save_path: str | Path | None = None) -> Any:
    if not has_torch():
        raise ImportError("torch not installed")
    import torch
    traced = torch.jit.trace(model, example)
    if save_path is not None:
        traced.save(str(save_path))
    return traced


__all__ = ["has_torch", "trace_module"]
