"""Round 538 — latency profiler."""

from __future__ import annotations


class TestLatency:
    def test_keys(self) -> None:
        from naviertwin.utils.latency_profiler import profile_call

        r = profile_call(lambda: sum(range(100)), n=20, warmup=2)
        for k in ["p50", "p90", "p99", "min", "max"]:
            assert k in r
            assert r[k] >= 0

    def test_min_le_max(self) -> None:
        from naviertwin.utils.latency_profiler import profile_call

        r = profile_call(lambda: 1, n=10, warmup=0)
        assert r["min"] <= r["max"]
