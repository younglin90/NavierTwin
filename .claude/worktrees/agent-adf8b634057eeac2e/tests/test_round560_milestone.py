"""Round 560 — FINAL milestone DD: industry scenarios (R551-R559) e2e."""

from __future__ import annotations


class TestMilestoneDD:
    def test_imports(self) -> None:
        from naviertwin.core.applied import (  # noqa: F401
            centrifugal_pump,
            chiller_cop,
            cyclone,
            ev_motor,
            fan_affinity,
            greenhouse,
            jensen_wake,
            mccabe_thiele,
            spray_smd,
        )

    def test_pump_wake_e2e(self) -> None:
        from naviertwin.core.applied.centrifugal_pump import operating_point
        from naviertwin.core.applied.jensen_wake import farm_velocity

        Q, H = operating_point(sys_a=2, sys_b=1, pump_a=10, pump_b=1)
        assert Q > 0 and H > 0

        farm = farm_velocity(V0=10, distances=[300, 300, 300], R=40)
        # monotone non-increasing along the row
        assert all(farm[i + 1] <= farm[i] + 1e-9 for i in range(len(farm) - 1))
