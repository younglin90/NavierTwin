"""Cross model: μ = μ_∞ + (μ_0 - μ_∞) / (1 + (k γ̇)^m).

Examples:
    >>> from naviertwin.core.rheology.cross_model import cross_viscosity
    >>> cross_viscosity(gamma_dot=0, mu_0=5, mu_inf=0.1, k=1, m=1)
    5.0
"""

from __future__ import annotations


def cross_viscosity(
    *, gamma_dot: float, mu_0: float, mu_inf: float, k: float = 1.0, m: float = 1.0,
) -> float:
    g = abs(float(gamma_dot))
    return float(mu_inf + (mu_0 - mu_inf) / (1.0 + (k * g) ** m))


__all__ = ["cross_viscosity"]
