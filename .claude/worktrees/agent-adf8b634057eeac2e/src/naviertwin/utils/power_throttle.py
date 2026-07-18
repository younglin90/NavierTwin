"""Power-aware throttling — reduce throughput if estimated power > budget.

Examples:
    >>> from naviertwin.utils.power_throttle import suggested_batch
    >>> suggested_batch(current_batch=64, watts_now=80, watt_budget=50)
    40
"""

from __future__ import annotations


def suggested_batch(
    *, current_batch: int, watts_now: float, watt_budget: float,
    min_batch: int = 1, max_batch: int = 1024,
) -> int:
    if watts_now <= watt_budget:
        return min(max_batch, current_batch + 1)
    ratio = watt_budget / max(watts_now, 1e-6)
    new_batch = max(min_batch, int(current_batch * ratio))
    return new_batch


__all__ = ["suggested_batch"]
