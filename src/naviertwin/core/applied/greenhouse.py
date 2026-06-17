"""Greenhouse energy balance — Q_solar - U A (T_in - T_out) = m c_p dT/dt.

Examples:
    >>> from naviertwin.core.applied.greenhouse import temperature_step
    >>> T = 25
    >>> temperature_step(T, T_out=10, Q_solar=500, U=5, A=10, m=200, cp=1000, dt=60)
"""

from __future__ import annotations

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required by greenhouse thermal analysis")


def temperature_step(
    T_in: float, *, T_out: float, Q_solar: float,
    U: float, A: float, m: float, cp: float, dt: float = 1.0,
) -> float:
    return float(
        _kernels.greenhouse_temperature_step(T_in, T_out, Q_solar, U, A, m, cp, dt),
    )


__all__ = ["temperature_step"]
