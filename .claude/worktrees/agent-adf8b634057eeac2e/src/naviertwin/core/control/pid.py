"""PID controller — 디지털 트윈 제어/보정 용.

u(t) = Kp e + Ki ∫e dt + Kd ė, anti-windup + derivative-on-measurement.

Examples:
    >>> from naviertwin.core.control.pid import PID
    >>> p = PID(kp=1.0, ki=0.1, kd=0.01)
    >>> u = p.step(setpoint=1.0, measurement=0.5, dt=0.1)
    >>> u > 0
    True
"""

from __future__ import annotations


class PID:
    def __init__(
        self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0,
        *, output_min: float | None = None, output_max: float | None = None,
        integral_min: float | None = None, integral_max: float | None = None,
        derivative_on_measurement: bool = True,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_min = output_min
        self.output_max = output_max
        self.integral_min = integral_min
        self.integral_max = integral_max
        self.dom = bool(derivative_on_measurement)
        self._integral = 0.0
        self._last_err = 0.0
        self._last_meas = 0.0
        self._first = True

    def reset(self) -> None:
        self._integral = 0.0
        self._last_err = 0.0
        self._last_meas = 0.0
        self._first = True

    def step(self, setpoint: float, measurement: float, dt: float) -> float:
        if dt <= 0:
            raise ValueError("dt > 0 required")
        err = setpoint - measurement
        # I
        self._integral += err * dt
        if self.integral_min is not None:
            self._integral = max(self._integral, self.integral_min)
        if self.integral_max is not None:
            self._integral = min(self._integral, self.integral_max)
        # D
        if self._first:
            d = 0.0
            self._first = False
        else:
            if self.dom:
                d = -(measurement - self._last_meas) / dt
            else:
                d = (err - self._last_err) / dt
        self._last_err = err
        self._last_meas = measurement
        u = self.kp * err + self.ki * self._integral + self.kd * d
        if self.output_min is not None:
            u = max(u, self.output_min)
        if self.output_max is not None:
            u = min(u, self.output_max)
        return float(u)


__all__ = ["PID"]
