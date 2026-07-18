"""Roe flux scheme — 1D Euler.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solvers.roe import roe_flux
    >>> UL = np.array([1.0, 0.0, 2.5])
    >>> UR = np.array([0.125, 0.0, 0.25])
    >>> F = roe_flux(UL, UR, gamma=1.4)
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
    H = (E + p) / max(rho, 1e-30)
    return rho, u, p, H


def _flux(U, gamma):
    rho, u, p, _ = _prim(U, gamma)
    return np.array([rho * u, rho * u * u + p, u * (U[2] + p)])


def roe_flux(
    UL: NDArray, UR: NDArray, *, gamma: float = 1.4,
) -> NDArray:
    rL, uL, pL, HL = _prim(UL, gamma)
    rR, uR, pR, HR = _prim(UR, gamma)
    sL = np.sqrt(rL)
    sR = np.sqrt(rR)
    u_avg = (sL * uL + sR * uR) / (sL + sR)
    H_avg = (sL * HL + sR * HR) / (sL + sR)
    a_avg = np.sqrt((gamma - 1) * max(H_avg - 0.5 * u_avg ** 2, 1e-30))
    eigs = np.array([abs(u_avg - a_avg), abs(u_avg), abs(u_avg + a_avg)])
    FL = _flux(UL, gamma)
    FR = _flux(UR, gamma)
    diss = 0.5 * eigs.max() * (UR - UL)
    return 0.5 * (FL + FR) - diss


__all__ = ["roe_flux"]
