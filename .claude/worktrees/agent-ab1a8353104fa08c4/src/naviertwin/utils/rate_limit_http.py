"""HTTP rate-limit middleware — token bucket per key.

Examples:
    >>> from naviertwin.utils.rate_limit_http import TokenBucket
    >>> bucket = TokenBucket(capacity=2, refill_per_sec=0.0)
    >>> bucket.allow("k"), bucket.allow("k"), bucket.allow("k")
    (True, True, False)
"""

from __future__ import annotations

import time


class TokenBucket:
    def __init__(self, *, capacity: float = 10.0, refill_per_sec: float = 1.0) -> None:
        self.capacity = capacity
        self.refill = refill_per_sec
        self._tokens: dict[str, float] = {}
        self._last: dict[str, float] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        prev_t = self._last.get(key, now)
        prev_tok = self._tokens.get(key, self.capacity)
        # refill
        tok = min(self.capacity, prev_tok + (now - prev_t) * self.refill)
        if tok >= 1.0:
            self._tokens[key] = tok - 1.0
            self._last[key] = now
            return True
        self._tokens[key] = tok
        self._last[key] = now
        return False


__all__ = ["TokenBucket"]
