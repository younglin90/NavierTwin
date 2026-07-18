"""Round 551 — centrifugal pump."""

from __future__ import annotations


class TestPump:
    def test_curve(self) -> None:
        from naviertwin.core.applied.centrifugal_pump import head_curve

        assert head_curve(Q=0, a=10, b=1) == 10.0
        assert head_curve(Q=2, a=10, b=1) == 6.0

    def test_op_point(self) -> None:
        from naviertwin.core.applied.centrifugal_pump import operating_point

        # pump 10 - Q²; system 2 + Q² → 10-Q² = 2+Q² → Q²=4 → Q=2, H=6
        Q, H = operating_point(sys_a=2, sys_b=1, pump_a=10, pump_b=1)
        assert abs(Q - 2.0) < 1e-9
        assert abs(H - 6.0) < 1e-9
