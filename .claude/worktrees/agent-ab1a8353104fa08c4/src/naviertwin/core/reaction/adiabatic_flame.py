"""Adiabatic flame temperature — energy balance lite.

T_ad = T_in + |ΔH_r| · n_fuel / (Σ n_prod · cp_avg).

Examples:
    >>> from naviertwin.core.reaction.adiabatic_flame import T_adiabatic
    >>> T_adiabatic(T_in=298, dHr=802000, n_fuel=1.0, n_products=10.0, cp_avg=40.0)
"""

from __future__ import annotations


def T_adiabatic(
    *, T_in: float, dHr: float, n_fuel: float, n_products: float, cp_avg: float,
) -> float:
    """ΔH_r in J/mol fuel; cp_avg in J/(mol·K)."""
    Q = dHr * n_fuel
    return float(T_in + Q / (n_products * cp_avg))


__all__ = ["T_adiabatic"]
