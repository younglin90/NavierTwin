"""Round 208 — dashboard aggregator."""

from __future__ import annotations


class TestDashboard:
    def test_push_summary(self) -> None:
        from naviertwin.core.monitoring.dashboard import DashboardAggregator

        d = DashboardAggregator(buffer_size=5)
        for v in [1.0, 2.0, 3.0]:
            d.push("loss", v)
        s = d.summary("loss")
        assert s["count"] == 3
        assert s["latest"] == 3.0
        assert s["mean"] == 2.0

    def test_ring(self) -> None:
        from naviertwin.core.monitoring.dashboard import DashboardAggregator

        d = DashboardAggregator(buffer_size=3)
        for v in [1, 2, 3, 4, 5]:
            d.push("m", float(v))
        s = d.summary("m")
        assert s["count"] == 3
        assert s["latest"] == 5.0

    def test_multi(self) -> None:
        from naviertwin.core.monitoring.dashboard import DashboardAggregator

        d = DashboardAggregator()
        d.push_many({"rmse": 0.1, "r2": 0.95})
        snap = d.snapshot()
        assert "rmse" in snap and "r2" in snap

    def test_reset(self) -> None:
        from naviertwin.core.monitoring.dashboard import DashboardAggregator

        d = DashboardAggregator()
        d.push("x", 1.0)
        d.reset("x")
        assert d.summary("x") == {"count": 0}
