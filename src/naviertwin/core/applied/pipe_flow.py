"""Pipe flow Darcy-Weisbach + Colebrook friction.

Examples:
    >>> from naviertwin.core.applied.pipe_flow import pressure_drop
    >>> pressure_drop(L=10, D=0.1, rho=1000, U=1.0, f=0.02)
    1000.0
"""

from __future__ import annotations

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def pressure_drop(*, L: float, D: float, rho: float, U: float, f: float) -> float:
    """Δp = f (L/D) (½ ρ U²)."""
    return f * (L / D) * 0.5 * rho * U * U


def friction_colebrook(*, Re: float, eps_over_D: float, n_iter: int = 50) -> float:
    """Iterate Colebrook: 1/√f = -2 log10(eps/3.7D + 2.51/(Re√f))."""
    return float(
        _kernels.friction_colebrook(float(Re), float(eps_over_D), int(n_iter)),
    )


__all__ = ["friction_colebrook", "pressure_drop"]
