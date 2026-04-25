"""Turbocharger matching — power balance compressor/turbine.

W_c = W_t (steady), find rpm equilibrium.

Examples:
    >>> from naviertwin.core.applied.turbo_match import match_rpm
    >>> rpm = match_rpm(comp_power=lambda n: n * 0.5, turb_power=lambda n: 1000 - 0.3 * n)
"""

from __future__ import annotations

from collections.abc import Callable


def match_rpm(
    comp_power: Callable[[float], float],
    turb_power: Callable[[float], float],
    *,
    rpm_min: float = 1000.0, rpm_max: float = 2.0e5,
    tol: float = 1.0,
) -> float:
    """Bisection on (W_t - W_c)."""
    lo, hi = rpm_min, rpm_max
    def f(n: float) -> float:
        return turb_power(n) - comp_power(n)
    if f(lo) * f(hi) > 0:
        return float(lo)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


__all__ = ["match_rpm"]
