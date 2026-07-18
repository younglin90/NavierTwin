"""경량 진행률/ETA 계산기 — tqdm 없이도 동작.

Examples:
    >>> from naviertwin.utils.progress import ProgressTracker
    >>> t = ProgressTracker(total=100)
    >>> t.update(10)
    >>> 0 < t.fraction < 1
    True
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ProgressTracker:
    """total 기준으로 진행량/속도/ETA 추정."""

    total: int
    _start: float = field(default_factory=time.monotonic, init=False)
    _done: int = field(default=0, init=False)
    _last_update: float = field(default_factory=time.monotonic, init=False)

    def update(self, delta: int = 1) -> None:
        self._done += int(delta)
        self._last_update = time.monotonic()

    @property
    def fraction(self) -> float:
        return 0.0 if self.total <= 0 else min(1.0, self._done / self.total)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    @property
    def rate(self) -> float:
        el = self.elapsed
        return 0.0 if el <= 0 else self._done / el

    @property
    def eta_seconds(self) -> float:
        r = self.rate
        left = self.total - self._done
        if r <= 0 or left <= 0:
            return 0.0
        return left / r

    def format(self) -> str:
        pct = self.fraction * 100.0
        return (
            f"{self._done}/{self.total} ({pct:5.1f}%) "
            f"rate={self.rate:7.2f}/s "
            f"elapsed={self.elapsed:6.1f}s "
            f"eta={self.eta_seconds:6.1f}s"
        )


__all__ = ["ProgressTracker"]
