"""Round 539 — power throttle."""

from __future__ import annotations


class TestPowerThrottle:
    def test_under_budget(self) -> None:
        from naviertwin.utils.power_throttle import suggested_batch

        assert suggested_batch(current_batch=10, watts_now=30, watt_budget=50) == 11

    def test_over_budget(self) -> None:
        from naviertwin.utils.power_throttle import suggested_batch

        assert suggested_batch(current_batch=64, watts_now=100, watt_budget=50) == 32
