"""Flakiness detector — track per-test pass/fail history.

Examples:
    >>> from naviertwin.utils.flakiness import FlakinessTracker
    >>> t = FlakinessTracker()
    >>> for r in [True, False, True, True]:
    ...     t.record('test_x', r)
    >>> 0 < t.flakiness('test_x') < 1
    True
"""

from __future__ import annotations

from collections import defaultdict


class FlakinessTracker:
    def __init__(self) -> None:
        self.history: dict[str, list[bool]] = defaultdict(list)

    def record(self, name: str, passed: bool) -> None:
        self.history[name].append(bool(passed))

    def flakiness(self, name: str) -> float:
        h = self.history.get(name, [])
        if len(h) < 2:
            return 0.0
        # transition rate (pass↔fail)
        flips = sum(1 for i in range(len(h) - 1) if h[i] != h[i + 1])
        return flips / (len(h) - 1)

    def flaky_tests(self, *, min_flakiness: float = 0.2) -> list[str]:
        return [n for n in self.history
                if self.flakiness(n) >= min_flakiness]


__all__ = ["FlakinessTracker"]
