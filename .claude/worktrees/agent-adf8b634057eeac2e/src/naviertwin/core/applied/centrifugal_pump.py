"""Centrifugal pump performance curve — H(Q) = a - b Q² (parabolic).

Examples:
    >>> from naviertwin.core.applied.centrifugal_pump import head_curve
    >>> head_curve(Q=2.0, a=20, b=2.0)
    12.0
"""

from __future__ import annotations


def head_curve(*, Q: float, a: float, b: float) -> float:
    return float(a - b * Q * Q)


def operating_point(*, sys_a: float, sys_b: float, pump_a: float, pump_b: float) -> tuple[float, float]:
    """System curve H_sys = sys_a + sys_b Q²; pump curve H_p = pump_a - pump_b Q².

    Intersect: pump_a - pump_b Q² = sys_a + sys_b Q²
    → Q² = (pump_a - sys_a) / (pump_b + sys_b)
    """
    num = pump_a - sys_a
    den = pump_b + sys_b
    if den <= 0 or num < 0:
        return 0.0, sys_a
    Q2 = num / den
    Q = float(Q2 ** 0.5)
    H = sys_a + sys_b * Q2
    return Q, float(H)


__all__ = ["head_curve", "operating_point"]
