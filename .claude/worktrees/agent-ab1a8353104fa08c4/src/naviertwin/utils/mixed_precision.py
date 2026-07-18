"""Mixed precision — torch.autocast wrapper (no-op if torch unavailable).

Examples:
    >>> from naviertwin.utils.mixed_precision import amp_context
    >>> with amp_context():
    ...     pass
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


@contextmanager
def amp_context(*, dtype: str = "float16", device: str = "cuda") -> Any:
    if not has_torch():
        yield None
        return
    import torch
    if not torch.cuda.is_available() or device == "cpu":
        yield None
        return
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    with torch.autocast(device_type="cuda", dtype=dt):
        yield


__all__ = ["amp_context", "has_torch"]
