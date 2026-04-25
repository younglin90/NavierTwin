"""Pipe flow Darcy-Weisbach + Colebrook friction.

Examples:
    >>> from naviertwin.core.applied.pipe_flow import pressure_drop
    >>> pressure_drop(L=10, D=0.1, rho=1000, U=1.0, f=0.02)
    1000.0
"""

from __future__ import annotations

import math


def pressure_drop(*, L: float, D: float, rho: float, U: float, f: float) -> float:
    """Δp = f (L/D) (½ ρ U²)."""
    return f * (L / D) * 0.5 * rho * U * U


def friction_colebrook(*, Re: float, eps_over_D: float, n_iter: int = 50) -> float:
    """Iterate Colebrook: 1/√f = -2 log10(eps/3.7D + 2.51/(Re√f))."""
    f = 0.02
    for _ in range(n_iter):
        rhs = -2 * math.log10(eps_over_D / 3.7 + 2.51 / (Re * math.sqrt(f) + 1e-30))
        f_new = 1.0 / (rhs * rhs)
        if abs(f_new - f) < 1e-10:
            return f_new
        f = f_new
    return f


__all__ = ["friction_colebrook", "pressure_drop"]
