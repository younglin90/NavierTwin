"""Entropy-stable flux — KEP (Kennedy-Gruber) for compressible Euler.

간단 form: F = (ρ̄ ū, ρ̄ ū² + p̄, ρ̄ ū H̄).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solvers.entropy_stable import kep_flux
    >>> UL = np.array([1.0, 0.5, 2.5])
    >>> UR = np.array([0.8, 0.4, 2.0])
    >>> F = kep_flux(UL, UR, gamma=1.4)
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


def kep_flux(
    UL: NDArray, UR: NDArray, *, gamma: float = 1.4,
) -> NDArray:
    rL, uL, pL, HL = _prim(UL, gamma)
    rR, uR, pR, HR = _prim(UR, gamma)
    rho_bar = 0.5 * (rL + rR)
    u_bar = 0.5 * (uL + uR)
    p_bar = 0.5 * (pL + pR)
    H_bar = 0.5 * (HL + HR)
    return np.array([
        rho_bar * u_bar,
        rho_bar * u_bar * u_bar + p_bar,
        rho_bar * u_bar * H_bar,
    ])


__all__ = ["kep_flux"]
