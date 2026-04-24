"""Gradient checkpointing wrapper — torch.utils.checkpoint.

Examples:
    >>> import pytest
    >>> torch = pytest.importorskip("torch")
    >>> from naviertwin.utils.grad_checkpoint import checkpoint_seq
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def checkpoint_seq(modules: list[Callable], x: Any) -> Any:
    """Apply each module in sequence under torch.utils.checkpoint."""
    if not has_torch():
        raise ImportError("torch not installed")
    import torch.utils.checkpoint as ckpt
    for m in modules:
        x = ckpt.checkpoint(m, x, use_reentrant=False)
    return x


__all__ = ["checkpoint_seq", "has_torch"]
