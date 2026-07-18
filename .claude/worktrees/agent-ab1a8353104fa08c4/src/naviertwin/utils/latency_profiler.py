"""Latency profiler — repeat callable, return percentile latencies (ms).

Examples:
    >>> from naviertwin.utils.latency_profiler import profile_call
    >>> r = profile_call(lambda: sum(range(100)), n=20)
    >>> 'p50' in r
    True
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


def profile_call(
    fn: Callable[[], Any], *, n: int = 50, warmup: int = 5,
) -> dict[str, float]:
    warmup_idx = 0
    while warmup_idx < warmup:
        fn()
        warmup_idx += 1
    samples = []
    sample_idx = 0
    while sample_idx < n:
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
        sample_idx += 1
    samples.sort()
    return {
        "p50": samples[n // 2],
        "p90": samples[int(n * 0.9)],
        "p99": samples[int(n * 0.99)],
        "min": samples[0],
        "max": samples[-1],
    }


__all__ = ["profile_call"]
