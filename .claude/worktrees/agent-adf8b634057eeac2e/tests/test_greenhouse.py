"""Round 558 — greenhouse."""

from __future__ import annotations


class TestGH:
    def test_steady(self) -> None:
        from naviertwin.core.applied.greenhouse import temperature_step

        T = 25.0
        for _ in range(20000):
            T = temperature_step(T, T_out=10, Q_solar=500, U=5, A=10,
                                    m=200, cp=1000, dt=10)
        # steady: T_out + Q_solar/(U A) = 10 + 500/50 = 20
        assert abs(T - 20.0) < 0.5
