"""Exponential backoff with jitter.

Examples:
    >>> from naviertwin.utils.backoff import backoff_delays
    >>> delays = backoff_delays(n=4, base=1.0, factor=2.0, jitter=0.0)
    >>> delays
    [1.0, 2.0, 4.0, 8.0]
"""

from __future__ import annotations

import random


def backoff_delays(
    *, n: int, base: float = 1.0, factor: float = 2.0, jitter: float = 0.1,
    cap: float = 60.0, seed: int | None = None,
) -> list[float]:
    rng = random.Random(seed)
    out = []
    d = base
    idx = 0
    while idx < n:
        j = rng.uniform(-jitter, jitter) * d if jitter > 0 else 0.0
        out.append(min(cap, max(0.0, d + j)))
        d *= factor
        idx += 1
    return out


__all__ = ["backoff_delays"]
