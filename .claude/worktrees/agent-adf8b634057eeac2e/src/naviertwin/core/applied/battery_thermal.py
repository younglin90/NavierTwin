"""Battery thermal — lumped capacitance: m c_p dT/dt = Q_gen - h A (T - T_amb).

Examples:
    >>> from naviertwin.core.applied.battery_thermal import temperature_step
    >>> T = 25.0
    >>> T = temperature_step(T, T_amb=20.0, Q_gen=10.0, h=5.0, A=0.1, m=0.5, cp=900, dt=1.0)
"""

from __future__ import annotations

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required by battery thermal analysis")


def temperature_step(
    T: float, *, T_amb: float, Q_gen: float, h: float, A: float,
    m: float, cp: float, dt: float = 1.0,
) -> float:
    return float(
        _kernels.battery_temperature_step(T, T_amb, Q_gen, h, A, m, cp, dt),
    )


def steady_temperature(*, T_amb: float, Q_gen: float, h: float, A: float) -> float:
    return float(_kernels.battery_steady_temperature(T_amb, Q_gen, h, A))


__all__ = ["steady_temperature", "temperature_step"]
