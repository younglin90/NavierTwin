"""간단한 벤치마크 러너 — warmup + 반복 + 평균/표준편차.

Examples:
    >>> from naviertwin.utils.benchmark import benchmark
    >>> r = benchmark(lambda: sum(range(1000)), n=10, warmup=2)
    >>> r["mean_ms"] > 0
    True
"""

from __future__ import annotations

import statistics
import time
from typing import Callable


def benchmark(
    fn: Callable, *, n: int = 20, warmup: int = 3,
) -> dict[str, float]:
    warmup_idx = 0
    while warmup_idx < warmup:
        fn()
        warmup_idx += 1
    times = []
    sample_idx = 0
    while sample_idx < n:
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
        sample_idx += 1
    return {
        "n": int(n),
        "mean_ms": float(statistics.mean(times)),
        "stdev_ms": float(statistics.stdev(times) if n > 1 else 0.0),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "median_ms": float(statistics.median(times)),
    }


def compare(
    cases: dict[str, Callable], *, n: int = 10, warmup: int = 2,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    items = list(cases.items())
    idx = 0
    while idx < len(items):
        name, fn = items[idx]
        results[name] = benchmark(fn, n=n, warmup=warmup)
        idx += 1
    return results


__all__ = ["benchmark", "compare"]
