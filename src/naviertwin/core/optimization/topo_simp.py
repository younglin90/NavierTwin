"""SIMP topology optimization 1D — minimize compliance s.t. volume frac.

E_i = E_min + ρ_i^p (E_max - E_min).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.topo_simp import simp_1d
    >>> rho = simp_1d(n=20, vol_frac=0.5, n_iter=20)
    >>> rho.shape, abs(rho.mean() - 0.5) < 0.05
    ((20,), True)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def simp_1d(
    n: int = 30,
    *,
    vol_frac: float = 0.5,
    p: float = 3.0,
    n_iter: int = 30,
    E_min: float = 1e-3,
    E_max: float = 1.0,
) -> NDArray[np.float64]:
    """1D bar with end load: solve K u = f, then update ρ via OC."""
    rho = np.full(n, vol_frac)
    f = np.zeros(n + 1)
    f[-1] = 1.0  # tip load
    for _ in range(n_iter):
        E = E_min + rho ** p * (E_max - E_min)
        # 1D bar stiffness: K (n+1, n+1), assemble per element
        K = np.zeros((n + 1, n + 1))
        for i in range(n):
            ke = E[i] * np.array([[1, -1], [-1, 1]])
            K[i:i + 2, i:i + 2] += ke
        # Dirichlet x_0 = 0
        Kr = K[1:, 1:]
        u = np.zeros(n + 1)
        u[1:] = np.linalg.solve(Kr, f[1:])
        # element compliance
        c_e = np.array([(u[i + 1] - u[i]) ** 2 * E[i] for i in range(n)])
        # OC update: dC/drho = -p ρ^{p-1} (E_max-E_min) (du/dx)²
        dc = -p * rho ** (p - 1) * (E_max - E_min) * (
            np.array([(u[i + 1] - u[i]) ** 2 for i in range(n)])
        )
        # bisection on lambda for volume constraint
        l1, l2 = 1e-9, 1e9
        while l2 - l1 > 1e-4:
            lm = 0.5 * (l1 + l2)
            rho_new = np.clip(rho * np.sqrt(-dc / lm + 0j).real, 1e-3, 1.0)
            if rho_new.mean() > vol_frac:
                l1 = lm
            else:
                l2 = lm
        rho = rho_new
        _ = c_e  # suppress
    return rho


__all__ = ["simp_1d"]
