"""What-if sandbox — apply parameter override without mutating base config.

Examples:
    >>> from naviertwin.core.twin.whatif import what_if
    >>> base = {'lr': 0.01, 'bs': 32}
    >>> def model(cfg): return cfg['lr'] * 100
    >>> what_if(base, {'lr': 0.05}, model)
    5.0
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def what_if(
    base_cfg: dict, overrides: dict, model_fn: Callable[[dict], Any],
) -> Any:
    cfg = {**base_cfg, **overrides}
    return model_fn(cfg)


__all__ = ["what_if"]
