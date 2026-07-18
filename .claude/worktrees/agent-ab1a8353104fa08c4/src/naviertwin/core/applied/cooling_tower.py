"""Cooling tower NTU — Merkel approximation.

ε = (T_h_in - T_h_out) / (T_h_in - T_wb).

Examples:
    >>> from naviertwin.core.applied.cooling_tower import tower_effectiveness
    >>> tower_effectiveness(T_h_in=40, T_h_out=30, T_wb=25)
    0.6666666666666666
"""

from __future__ import annotations


def tower_effectiveness(*, T_h_in: float, T_h_out: float, T_wb: float) -> float:
    denom = T_h_in - T_wb
    if denom <= 0:
        return 0.0
    return float((T_h_in - T_h_out) / denom)


def NTU_from_effectiveness(*, eps: float) -> float:
    """For Cr=0 (counter-flow with phase change): ε = 1 - exp(-NTU)."""
    if eps >= 1.0:
        return float("inf")
    if eps <= 0:
        return 0.0
    import math
    return float(-math.log(1 - eps))


__all__ = ["NTU_from_effectiveness", "tower_effectiveness"]
