"""유체역학 무차원수 계산기 — Re / Ma / Pr / Nu / Fr / We / Ra / Pe.

Examples:
    >>> from naviertwin.core.analysis.dimensionless import reynolds, mach
    >>> reynolds(U=10.0, L=0.1, nu=1e-5)
    100000.0
    >>> round(mach(U=34.0, a=340.0), 2)
    0.1
"""

from __future__ import annotations


def reynolds(U: float, L: float, nu: float) -> float:
    """Re = U·L/ν."""
    return U * L / nu


def reynolds_rho(U: float, L: float, rho: float, mu: float) -> float:
    """Re = ρ·U·L/μ."""
    return rho * U * L / mu


def mach(U: float, a: float) -> float:
    """Ma = U/a (a=음속)."""
    return U / a


def prandtl(cp: float, mu: float, k: float) -> float:
    """Pr = cp·μ/k."""
    return cp * mu / k


def nusselt(h: float, L: float, k: float) -> float:
    """Nu = h·L/k."""
    return h * L / k


def froude(U: float, L: float, g: float = 9.81) -> float:
    """Fr = U/√(gL)."""
    from math import sqrt
    return U / sqrt(g * L)


def weber(rho: float, U: float, L: float, sigma: float) -> float:
    """We = ρU²L/σ."""
    return rho * U * U * L / sigma


def peclet(U: float, L: float, alpha: float) -> float:
    """Pe = U·L/α."""
    return U * L / alpha


def rayleigh(
    g: float, beta: float, dT: float, L: float, nu: float, alpha: float,
) -> float:
    """Ra = g·β·ΔT·L³/(ν·α)."""
    return g * beta * dT * (L ** 3) / (nu * alpha)


def strouhal(f: float, L: float, U: float) -> float:
    """St = f·L/U (vortex shedding)."""
    return f * L / U


__all__ = [
    "reynolds", "reynolds_rho", "mach", "prandtl", "nusselt",
    "froude", "weber", "peclet", "rayleigh", "strouhal",
]
