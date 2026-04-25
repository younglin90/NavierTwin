"""Round 546 — SLO."""

from __future__ import annotations


class TestSLO:
    def test_at_budget(self) -> None:
        from naviertwin.core.twin.slo import burn_rate

        # 1% errors with 99% SLO → burn rate 1.0
        assert abs(burn_rate(error_count=10, total_count=1000, slo=0.99) - 1.0) < 1e-9

    def test_under_budget(self) -> None:
        from naviertwin.core.twin.slo import burn_rate

        assert burn_rate(error_count=1, total_count=1000, slo=0.99) < 1.0
