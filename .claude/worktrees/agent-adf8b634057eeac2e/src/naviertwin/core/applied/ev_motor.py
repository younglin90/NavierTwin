"""EV motor torque/speed envelope — constant T below ω_base, P=Tω above.

Examples:
    >>> from naviertwin.core.applied.ev_motor import torque_envelope
    >>> torque_envelope(omega=100, omega_base=300, T_max=200, P_max=60000)
    200
"""

from __future__ import annotations


def torque_envelope(*, omega: float, omega_base: float, T_max: float, P_max: float) -> float:
    if omega <= 0:
        return T_max
    if omega <= omega_base:
        return T_max
    return min(T_max, P_max / omega)


def power_envelope(*, omega: float, omega_base: float, T_max: float, P_max: float) -> float:
    if omega <= omega_base:
        return omega * T_max
    return min(P_max, omega * T_max)


__all__ = ["power_envelope", "torque_envelope"]
