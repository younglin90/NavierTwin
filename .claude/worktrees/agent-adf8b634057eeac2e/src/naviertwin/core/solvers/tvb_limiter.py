"""TVB (Total Variation Bounded) limiter — Shu 1987.

Allows local extrema if |Δ| < M h².

Examples:
    >>> from naviertwin.core.solvers.tvb_limiter import tvb_minmod
    >>> tvb_minmod(0.05, 1.0, 2.0, M=10.0, h=0.1)
    0.05
"""

from __future__ import annotations


def tvb_minmod(a: float, b: float, c: float, *, M: float, h: float) -> float:
    """If |a| ≤ M h² → return a, else minmod(a, b, c)."""
    if abs(a) <= M * h * h:
        return a
    if (a > 0 and b > 0 and c > 0) or (a < 0 and b < 0 and c < 0):
        return min(abs(a), abs(b), abs(c)) * (1.0 if a > 0 else -1.0)
    return 0.0


__all__ = ["tvb_minmod"]
