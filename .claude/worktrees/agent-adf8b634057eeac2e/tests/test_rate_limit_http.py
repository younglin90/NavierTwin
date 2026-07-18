"""Round 359 — rate limit."""

from __future__ import annotations


class TestRateLimit:
    def test_burst_then_block(self) -> None:
        from naviertwin.utils.rate_limit_http import TokenBucket

        b = TokenBucket(capacity=3, refill_per_sec=0.0)
        assert b.allow("k") is True
        assert b.allow("k") is True
        assert b.allow("k") is True
        assert b.allow("k") is False

    def test_per_key(self) -> None:
        from naviertwin.utils.rate_limit_http import TokenBucket

        b = TokenBucket(capacity=1, refill_per_sec=0.0)
        assert b.allow("a")
        assert not b.allow("a")
        # different key has independent bucket
        assert b.allow("b")
