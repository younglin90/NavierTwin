"""무차원수 계산 모듈.

Nusselt, Reynolds, Prandtl 등 CFD 후처리에 자주 쓰이는 무차원수 공식.

Formulas:
    Re = ρ·U·L / μ
    Pr = ν / α = μ·cp / k
    Nu = h·L / k

Examples:
    >>> from naviertwin.core.flow_analysis.thermofluids.nondim import (
    ...     reynolds, prandtl, nusselt,
    ... )
    >>> Re = reynolds(rho=1.225, U=10.0, L=1.0, mu=1.8e-5)
    >>> Pr = prandtl(mu=1.8e-5, cp=1005.0, k=0.026)
    >>> Nu = nusselt(h=25.0, L=1.0, k=0.026)
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import NDArray

Number = Union[float, NDArray[np.float64]]


def reynolds(rho: Number, U: Number, L: Number, mu: Number) -> Number:
    """Re = ρ·U·L / μ."""
    return np.asarray(rho) * np.asarray(U) * np.asarray(L) / np.asarray(mu)


def prandtl(mu: Number, cp: Number, k: Number) -> Number:
    """Pr = μ·cp / k."""
    return np.asarray(mu) * np.asarray(cp) / np.asarray(k)


def nusselt(h: Number, L: Number, k: Number) -> Number:
    """Nu = h·L / k."""
    return np.asarray(h) * np.asarray(L) / np.asarray(k)


def peclet(re: Number, pr: Number) -> Number:
    """Pe = Re · Pr."""
    return np.asarray(re) * np.asarray(pr)


def grashof(rho: Number, g: float, beta: Number, dT: Number, L: Number, mu: Number) -> Number:
    """Gr = g·β·ΔT·L³·ρ² / μ²."""
    rho_a = np.asarray(rho)
    return g * np.asarray(beta) * np.asarray(dT) * np.asarray(L) ** 3 * rho_a**2 / np.asarray(mu) ** 2


def rayleigh(gr: Number, pr: Number) -> Number:
    """Ra = Gr · Pr."""
    return np.asarray(gr) * np.asarray(pr)


__all__ = ["reynolds", "prandtl", "nusselt", "peclet", "grashof", "rayleigh"]
