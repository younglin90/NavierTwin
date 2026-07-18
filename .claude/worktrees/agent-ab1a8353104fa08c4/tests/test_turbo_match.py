"""Round 455 — turbo match."""

from __future__ import annotations


class TestTurboMatch:
    def test_balance(self) -> None:
        from naviertwin.core.applied.turbo_match import match_rpm

        # comp = 0.5 n, turb = 1000 - 0.3 n → equality at n = 1250
        rpm = match_rpm(
            comp_power=lambda n: 0.5 * n,
            turb_power=lambda n: 1000 - 0.3 * n,
            rpm_min=0, rpm_max=10000, tol=0.5,
        )
        assert abs(rpm - 1250.0) < 5.0
