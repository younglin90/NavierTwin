"""Time/compute budget tracker.

Examples:
    >>> from naviertwin.utils.workflow.budget import BudgetTracker
    >>> b = BudgetTracker(max_seconds=10, max_steps=100)
    >>> b.tick(); b.exceeded()
    False
"""

from __future__ import annotations

import time


class BudgetTracker:
    def __init__(self, *, max_seconds: float | None = None,
                 max_steps: int | None = None) -> None:
        self.max_seconds = max_seconds
        self.max_steps = max_steps
        self.t0 = time.monotonic()
        self.steps = 0

    def tick(self) -> None:
        self.steps += 1

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def exceeded(self) -> bool:
        if self.max_seconds is not None and self.elapsed() > self.max_seconds:
            return True
        if self.max_steps is not None and self.steps >= self.max_steps:
            return True
        return False


__all__ = ["BudgetTracker"]
