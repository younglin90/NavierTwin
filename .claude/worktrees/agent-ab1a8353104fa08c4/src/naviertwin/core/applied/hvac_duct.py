"""HVAC duct loss — friction + fittings (K-coef).

Examples:
    >>> from naviertwin.core.applied.hvac_duct import total_pressure_loss
    >>> total_pressure_loss(L=10, D=0.3, rho=1.2, U=5, f=0.02, K_total=2.0)
"""

from __future__ import annotations


def total_pressure_loss(
    *, L: float, D: float, rho: float, U: float,
    f: float = 0.02, K_total: float = 0.0,
) -> float:
    """Δp = (f L/D + K) ½ ρ U²."""
    return (f * L / D + K_total) * 0.5 * rho * U * U


def duct_velocity(*, mdot: float, rho: float, A: float) -> float:
    return mdot / (rho * A)


__all__ = ["duct_velocity", "total_pressure_loss"]
