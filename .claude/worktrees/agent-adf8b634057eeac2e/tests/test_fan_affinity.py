"""Round 552 — fan affinity."""

from __future__ import annotations


class TestFan:
    def test_double_speed(self) -> None:
        from naviertwin.core.applied.fan_affinity import scale_Q_H_P

        # 1000 → 2000 RPM: Q×2, H×4, P×8
        Q, H, P = scale_Q_H_P(Q1=1, H1=1, P1=1, N1=1000, N2=2000)
        assert Q == 2.0
        assert H == 4.0
        assert P == 8.0
