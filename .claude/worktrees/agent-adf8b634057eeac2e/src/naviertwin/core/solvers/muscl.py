"""MUSCL slope limiters — minmod, van Leer, superbee.

Examples:
    >>> from naviertwin.core.solvers.muscl import minmod, van_leer
    >>> minmod(1.0, 2.0)
    1.0
    >>> minmod(1.0, -2.0)
    0.0
"""

from __future__ import annotations


def minmod(a: float, b: float) -> float:
    if a * b <= 0:
        return 0.0
    return a if abs(a) < abs(b) else b


def van_leer(a: float, b: float) -> float:
    if a * b <= 0:
        return 0.0
    return 2 * a * b / (a + b)


def superbee(a: float, b: float) -> float:
    if a * b <= 0:
        return 0.0
    s = 1.0 if a > 0 else -1.0
    return s * max(min(2 * abs(a), abs(b)), min(abs(a), 2 * abs(b)))


__all__ = ["minmod", "superbee", "van_leer"]
