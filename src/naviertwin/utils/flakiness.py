"""Flakiness detector — track per-test pass/fail history.

Examples:
    >>> from naviertwin.utils.flakiness import FlakinessTracker
    >>> t = FlakinessTracker()
    >>> t.record('test_x', True)
    >>> t.record('test_x', False)
    >>> t.record('test_x', True)
    >>> t.record('test_x', True)
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
        flips = 0
        idx = 0
        last_idx = len(h) - 1
        while idx < last_idx:
            if h[idx] != h[idx + 1]:
                flips += 1
            idx += 1
        return flips / (len(h) - 1)

    def flaky_tests(self, *, min_flakiness: float = 0.2) -> list[str]:
        names = list(self.history)
        out: list[str] = []
        idx = 0
        while idx < len(names):
            n = names[idx]
            if self.flakiness(n) >= min_flakiness:
                out.append(n)
            idx += 1
        return out


__all__ = ["FlakinessTracker"]
