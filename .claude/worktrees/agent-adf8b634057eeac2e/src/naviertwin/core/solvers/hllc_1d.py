"""HLLC Riemann solver — 1D Euler equations (Toro).

State U = (ρ, ρu, E)ᵀ. γ-법칙 ideal gas.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solvers.hllc_1d import hllc_flux
    >>> UL = np.array([1.0, 0.0, 2.5])
    >>> UR = np.array([0.125, 0.0, 0.25])
    >>> F = hllc_flux(UL, UR, gamma=1.4)
    >>> F.shape
    (3,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _prim(U, gamma):
    rho = U[0]
    u = U[1] / max(rho, 1e-30)
    E = U[2]
    p = (gamma - 1.0) * (E - 0.5 * rho * u * u)
    return rho, u, p


def _flux(U, gamma):
    rho, u, p = _prim(U, gamma)
    return np.array([rho * u, rho * u * u + p, u * (U[2] + p)])


def hllc_flux(
    UL: NDArray[np.float64], UR: NDArray[np.float64], *, gamma: float = 1.4,
) -> NDArray[np.float64]:
    rL, uL, pL = _prim(UL, gamma)
    rR, uR, pR = _prim(UR, gamma)
    aL = np.sqrt(gamma * pL / max(rL, 1e-30))
    aR = np.sqrt(gamma * pR / max(rR, 1e-30))
    SL = min(uL, uR) - max(aL, aR)
    SR = max(uL, uR) + max(aL, aR)
    Sstar = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / (
        rL * (SL - uL) - rR * (SR - uR) + 1e-30
    )
    FL = _flux(UL, gamma)
    FR = _flux(UR, gamma)
    if SL >= 0:
        return FL
    if SR <= 0:
        return FR
    if Sstar >= 0:
        UstarL = rL * (SL - uL) / (SL - Sstar) * np.array([
            1.0, Sstar,
            UL[2] / max(rL, 1e-30) + (Sstar - uL) * (Sstar + pL / (rL * (SL - uL))),
        ])
        return FL + SL * (UstarL - UL)
    UstarR = rR * (SR - uR) / (SR - Sstar) * np.array([
        1.0, Sstar,
        UR[2] / max(rR, 1e-30) + (Sstar - uR) * (Sstar + pR / (rR * (SR - uR))),
    ])
    return FR + SR * (UstarR - UR)


__all__ = ["hllc_flux"]
