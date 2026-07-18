"""Round 347 — Adaptive PID."""

from __future__ import annotations


class TestAdaptivePID:
    def test_gain_schedule(self) -> None:
        from naviertwin.core.control.adaptive_pid import AdaptivePID

        pid = AdaptivePID(schedule=lambda m: (2.0, 0.0, 0.0))
        u = pid.step(setpoint=1.0, measured=0.5, dt=1.0)
        assert abs(u - 1.0) < 1e-12  # 2 * 0.5

    def test_integral_accumulates(self) -> None:
        from naviertwin.core.control.adaptive_pid import AdaptivePID

        pid = AdaptivePID(schedule=lambda m: (0.0, 1.0, 0.0))
        pid.step(setpoint=1.0, measured=0.0, dt=1.0)  # integral += 1
        u = pid.step(setpoint=1.0, measured=0.0, dt=1.0)  # integral += 1 → 2
        assert abs(u - 2.0) < 1e-12

    def test_simulate_first_order(self) -> None:
        from naviertwin.core.control.adaptive_pid import AdaptivePID

        pid = AdaptivePID(schedule=lambda m: (1.0, 0.5, 0.0))
        x = 0.0
        for _ in range(100):
            u = pid.step(setpoint=1.0, measured=x, dt=0.1)
            x = x + 0.1 * (u - 0.5 * x)
        assert abs(x - 1.0) < 0.1
