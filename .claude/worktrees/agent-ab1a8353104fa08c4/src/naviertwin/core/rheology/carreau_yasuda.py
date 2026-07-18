"""Carreau-Yasuda model: μ = μ_∞ + (μ_0 - μ_∞)(1 + (λ γ̇)^a)^((n-1)/a).

Examples:
    >>> from naviertwin.core.rheology.carreau_yasuda import cy_viscosity
    >>> cy_viscosity(gamma_dot=0, mu_0=10, mu_inf=1, lam=1, a=2, n=0.5)
    10.0
"""

from __future__ import annotations


def cy_viscosity(
    *, gamma_dot: float, mu_0: float, mu_inf: float,
    lam: float = 1.0, a: float = 2.0, n: float = 0.5,
) -> float:
    g = abs(float(gamma_dot))
    return float(mu_inf + (mu_0 - mu_inf) * (1.0 + (lam * g) ** a) ** ((n - 1) / a))


__all__ = ["cy_viscosity"]
