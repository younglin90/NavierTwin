"""Round 97 — 토큰 버킷 rate limiter."""

from __future__ import annotations

import time

import pytest


class TestTokenBucket:
    def test_burst(self) -> None:
        from naviertwin.utils.rate_limit import TokenBucket

        b = TokenBucket(capacity=5, refill_per_sec=100.0)
        for _ in range(5):
            assert b.try_acquire() is True
        assert b.try_acquire() is False

    def test_refill(self) -> None:
        from naviertwin.utils.rate_limit import TokenBucket

        b = TokenBucket(capacity=2, refill_per_sec=200.0)
        assert b.try_acquire(2) is True
        assert b.try_acquire() is False
        time.sleep(0.05)  # ≥ 10 tokens refilled, but capped at 2
        assert b.try_acquire() is True

    def test_acquire_timeout(self) -> None:
        from naviertwin.utils.rate_limit import TokenBucket

        b = TokenBucket(capacity=1, refill_per_sec=1.0)
        assert b.try_acquire() is True
        assert b.acquire(timeout=0.05) is False

    def test_invalid(self) -> None:
        from naviertwin.utils.rate_limit import TokenBucket

        with pytest.raises(ValueError):
            TokenBucket(capacity=0)
        with pytest.raises(ValueError):
            TokenBucket(refill_per_sec=0)
