"""Mixture fraction Z = (β - β_ox) / (β_fuel - β_ox), Bilger 1989.

Examples:
    >>> from naviertwin.core.reaction.mixture_fraction import bilger_Z
    >>> bilger_Z(beta=0.5, beta_fuel=1.0, beta_ox=0.0)
    0.5
"""

from __future__ import annotations


def bilger_Z(*, beta: float, beta_fuel: float, beta_ox: float) -> float:
    denom = beta_fuel - beta_ox
    if abs(denom) < 1e-30:
        return 0.0
    return float((beta - beta_ox) / denom)


__all__ = ["bilger_Z"]
