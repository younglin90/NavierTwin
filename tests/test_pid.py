"""Round 197 — PID."""

from __future__ import annotations

import pytest


class TestPID:
    def test_proportional_response(self) -> None:
        from naviertwin.core.control.pid import PID

        p = PID(kp=2.0)
        u = p.step(setpoint=1.0, measurement=0.0, dt=0.1)
        assert u == pytest.approx(2.0)

    def test_integral_accumulates(self) -> None:
        from naviertwin.core.control.pid import PID

        p = PID(kp=0.0, ki=1.0)
        _ = p.step(1.0, 0.0, dt=0.5)
        _ = p.step(1.0, 0.0, dt=0.5)
        u = p.step(1.0, 0.0, dt=0.5)
        assert u == pytest.approx(1.5)

    def test_saturation(self) -> None:
        from naviertwin.core.control.pid import PID

        p = PID(kp=100.0, output_max=1.0)
        u = p.step(1.0, 0.0, dt=0.1)
        assert u == 1.0

    def test_plant_regulation(self) -> None:
        """1차 plant y_{k+1} = 0.9 y_k + b u 제어."""
        from naviertwin.core.control.pid import PID

        p = PID(kp=1.0, ki=0.5, kd=0.05)
        y = 0.0
        for _ in range(200):
            u = p.step(setpoint=1.0, measurement=y, dt=0.1)
            y = 0.9 * y + 0.1 * u
        assert abs(y - 1.0) < 0.05

    def test_invalid_dt(self) -> None:
        from naviertwin.core.control.pid import PID

        with pytest.raises(ValueError):
            PID().step(1.0, 0.0, dt=0.0)

    def test_reset(self) -> None:
        from naviertwin.core.control.pid import PID

        p = PID(kp=0.0, ki=1.0)
        p.step(1.0, 0.0, 0.5)
        p.reset()
        u = p.step(1.0, 0.0, 0.5)
        assert u == pytest.approx(0.5)  # integral reset → only ki * (0.5 * 1)
