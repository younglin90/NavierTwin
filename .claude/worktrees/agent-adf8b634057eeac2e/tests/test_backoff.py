"""Round 448 — backoff."""

from __future__ import annotations


class TestBackoff:
    def test_no_jitter(self) -> None:
        from naviertwin.utils.backoff import backoff_delays

        d = backoff_delays(n=4, base=1.0, factor=2.0, jitter=0.0)
        assert d == [1.0, 2.0, 4.0, 8.0]

    def test_cap(self) -> None:
        from naviertwin.utils.backoff import backoff_delays

        d = backoff_delays(n=10, base=1.0, factor=10.0, jitter=0.0, cap=5.0)
        assert max(d) <= 5.0
