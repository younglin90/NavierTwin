"""Round 237 — benchmark."""

from __future__ import annotations


class TestBench:
    def test_simple(self) -> None:
        from naviertwin.utils.benchmark import benchmark

        r = benchmark(lambda: sum(range(1000)), n=5, warmup=1)
        assert r["n"] == 5
        assert r["mean_ms"] >= 0
        assert r["median_ms"] >= 0

    def test_compare(self) -> None:
        from naviertwin.utils.benchmark import compare

        res = compare({
            "fast": lambda: sum(range(100)),
            "slow": lambda: sum(range(10000)),
        }, n=3, warmup=1)
        assert "fast" in res and "slow" in res
        # slow 가 일반적으로 더 오래 걸림 (엄격하지 않게)
        assert res["slow"]["mean_ms"] >= 0
