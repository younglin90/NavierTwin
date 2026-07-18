"""AUSM+ flux scheme — Liou 1996, 1D Euler.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solvers.ausm_plus import ausm_plus_flux
    >>> UL = np.array([1.0, 0.0, 2.5])
    >>> UR = np.array([0.125, 0.0, 0.25])
    >>> F = ausm_plus_flux(UL, UR, gamma=1.4)
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
    a = np.sqrt(gamma * max(p, 1e-30) / max(rho, 1e-30))
    return rho, u, p, H, a


def _M_split(M, sign):
    if abs(M) >= 1.0:
        return 0.5 * (M + sign * abs(M))
    return sign * 0.25 * (M + sign) ** 2


def _P_split(M, sign):
    if abs(M) >= 1.0:
        return 0.5 * (1 + sign * np.sign(M))
    return 0.25 * (M + sign) ** 2 * (2 - sign * M)


def ausm_plus_flux(
    UL: NDArray[np.float64], UR: NDArray[np.float64], *, gamma: float = 1.4,
) -> NDArray[np.float64]:
    rL, uL, pL, HL, aL = _prim(UL, gamma)
    rR, uR, pR, HR, aR = _prim(UR, gamma)
    a_iface = 0.5 * (aL + aR)
    ML = uL / a_iface
    MR = uR / a_iface
    M_half = _M_split(ML, +1) + _M_split(MR, -1)
    p_half = _P_split(ML, +1) * pL + _P_split(MR, -1) * pR
    if M_half >= 0:
        Phi = np.array([rL, rL * uL, rL * HL])
    else:
        Phi = np.array([rR, rR * uR, rR * HR])
    F = a_iface * M_half * Phi + np.array([0.0, p_half, 0.0])
    return F


__all__ = ["ausm_plus_flux"]
