"""Adaptive PID — gain scheduling on operating point.

Examples:
    >>> from naviertwin.core.control.adaptive_pid import AdaptivePID
    >>> pid = AdaptivePID(schedule=lambda x: (1.0, 0.1, 0.0))
    >>> u = pid.step(setpoint=1.0, measured=0.0, dt=0.1)
"""

from __future__ import annotations

from collections.abc import Callable


class AdaptivePID:
    def __init__(
        self,
        schedule: Callable[[float], tuple[float, float, float]],
    ) -> None:
        """schedule: setpoint or measured → (Kp, Ki, Kd)."""
        self.schedule = schedule
        self.integral = 0.0
        self.prev_err = 0.0

    def step(
        self, *, setpoint: float, measured: float, dt: float = 1.0,
    ) -> float:
        Kp, Ki, Kd = self.schedule(measured)
        err = setpoint - measured
        self.integral += err * dt
        deriv = (err - self.prev_err) / max(dt, 1e-12)
        self.prev_err = err
        return Kp * err + Ki * self.integral + Kd * deriv

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_err = 0.0


__all__ = ["AdaptivePID"]
