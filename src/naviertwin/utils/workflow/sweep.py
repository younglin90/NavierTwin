"""Hyperparameter sweep — grid + random.

Examples:
    >>> from naviertwin.utils.workflow.sweep import grid_sweep
    >>> list(grid_sweep({'lr': [0.1, 0.01], 'bs': [32]}))
    [{'lr': 0.1, 'bs': 32}, {'lr': 0.01, 'bs': 32}]
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product
from typing import Any

import numpy as np


def grid_sweep(space: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    keys = list(space.keys())
    value_lists: list[list[Any]] = []
    key_idx = 0
    while key_idx < len(keys):
        value_lists.append(space[keys[key_idx]])
        key_idx += 1
    combos = list(product(*value_lists))
    combo_idx = 0
    while combo_idx < len(combos):
        combo = combos[combo_idx]
        yield dict(zip(keys, combo, strict=True))
        combo_idx += 1


def random_sweep(
    space: dict[str, tuple[float, float]], n: int, *, seed: int = 0,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    out = []
    items = list(space.items())
    sample_idx = 0
    while sample_idx < n:
        sample: dict[str, float] = {}
        item_idx = 0
        while item_idx < len(items):
            k, (lo, hi) = items[item_idx]
            sample[k] = float(rng.uniform(lo, hi))
            item_idx += 1
        out.append(sample)
        sample_idx += 1
    return out


__all__ = ["grid_sweep", "random_sweep"]
