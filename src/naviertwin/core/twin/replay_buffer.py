"""Twin replay buffer — bounded ring buffer of (timestamp, state) tuples.

Examples:
    >>> from naviertwin.core.twin.replay_buffer import ReplayBuffer
    >>> r = ReplayBuffer(capacity=3)
    >>> r.add(0, 0); r.add(1, 2); r.add(2, 4); r.add(3, 6); r.add(4, 8)
    >>> len(r)
    3
"""

from __future__ import annotations

from collections import deque
from typing import Any


class ReplayBuffer:
    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = int(capacity)
        self.buf: deque[tuple[float, Any]] = deque(maxlen=self.capacity)

    def add(self, t: float, state: Any) -> None:
        self.buf.append((t, state))

    def __len__(self) -> int:
        return len(self.buf)

    def latest(self, n: int = 1) -> list[tuple[float, Any]]:
        return list(self.buf)[-n:]

    def all(self) -> list[tuple[float, Any]]:
        return list(self.buf)


__all__ = ["ReplayBuffer"]
