"""Pipeline parallel scheduler (GPipe-style microbatch schedule, toy).

Examples:
    >>> from naviertwin.utils.pipeline_parallel import gpipe_schedule
    >>> gpipe_schedule(n_stages=3, n_microbatch=2)
    [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (2, 1)]
"""

from __future__ import annotations


def gpipe_schedule(*, n_stages: int, n_microbatch: int) -> list[tuple[int, int]]:
    """Returns list of (stage, microbatch_idx) in execution order."""
    schedule = []
    t = 0
    last_t = n_stages + n_microbatch - 1
    while t < last_t:
        s = 0
        while s < n_stages:
            mb = t - s
            if 0 <= mb < n_microbatch:
                schedule.append((s, mb))
            s += 1
        t += 1
    return schedule


__all__ = ["gpipe_schedule"]
