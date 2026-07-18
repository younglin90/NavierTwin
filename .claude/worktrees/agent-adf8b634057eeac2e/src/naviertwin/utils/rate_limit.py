"""간단한 rate limiter — 토큰 버킷.

GUI 이벤트 스로틀, 외부 API 호출 제한 등.

Examples:
    >>> from naviertwin.utils.rate_limit import TokenBucket
    >>> b = TokenBucket(capacity=5, refill_per_sec=10)
    >>> b.try_acquire()
    True
"""

from __future__ import annotations

import time
from threading import Lock


class TokenBucket:
    """고정 용량 토큰 버킷."""

    def __init__(self, capacity: int = 10, refill_per_sec: float = 1.0) -> None:
        if capacity < 1 or refill_per_sec <= 0:
            raise ValueError("capacity >= 1, refill_per_sec > 0")
        self.capacity = int(capacity)
        self.refill = float(refill_per_sec)
        self._tokens: float = float(capacity)
        self._last = time.monotonic()
        self._lock = Lock()

    def _refill_now(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill)
        self._last = now

    def try_acquire(self, n: int = 1) -> bool:
        with self._lock:
            self._refill_now()
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False

    def acquire(self, n: int = 1, timeout: float | None = None) -> bool:
        """blocking 획득. timeout=None 이면 무제한 대기."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self.try_acquire(n):
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(max(0.005, n / self.refill / 10))

    @property
    def tokens(self) -> float:
        with self._lock:
            self._refill_now()
            return self._tokens


__all__ = ["TokenBucket"]
