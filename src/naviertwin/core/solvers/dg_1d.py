"""1D DG - cell coefficient arrays plus local Lagrange nodes (GLL).

Advection ∂u/∂t + c ∂u/∂x = 0 (스칼라 상수 c, upwind flux).

Examples:
    >>> from naviertwin.core.solvers.dg_1d import solve_advection_1d_dg
    >>> import numpy as np
    >>> x, t, U = solve_advection_1d_dg(n_cells=8, T=0.2)
    >>> U.shape[0] > 0
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def _lobatto_nodes(p: int) -> NDArray[np.float64]:
    """Gauss-Lobatto-Legendre nodes on [-1, 1]."""
    if p == 1:
        return np.array([-1.0, 1.0])
    from numpy.polynomial.legendre import legder, legroots
    poly = np.zeros(p + 1)
    poly[-1] = 1.0
    dpoly = legder(poly)
    inner = legroots(dpoly)
    return np.sort(np.r_[-1.0, inner, 1.0])


def solve_advection_1d_dg(
    n_cells: int = 16, L: float = 1.0, T: float = 0.2,
    c: float = 1.0,
    u0: Callable | None = None,
    *, p: int = 2, cfl: float = 0.3,
) -> tuple[NDArray, NDArray, NDArray]:
    """Pk DG advection (periodic). Returns (x_all, t, U[n_nodes, n_steps+1])."""
    xi = _lobatto_nodes(p)
    n_local = xi.size
    h = L / n_cells
    # cell centers plus local positions
    centers = (np.arange(n_cells, dtype=np.float64) + 0.5) * h
    x_all = (centers[:, None] + 0.5 * h * xi[None, :]).reshape(-1)
    # mass/stiffness per cell (reference)
    # Use Lagrange basis at GLL points; mass matrix = diag(w) (lumped)
    # weights via Gauss-Lobatto formulae (trapezoidal-like)
    # Lumped mass with fixed GL weights approximation.
    from numpy.polynomial.legendre import legval
    # compute lumped weights: w_i = 2 / (p(p+1) [P_p(xi_i)]²)
    Pp = np.zeros(p + 1)
    Pp[-1] = 1.0
    w = 2.0 / (p * (p + 1) * legval(xi, Pp) ** 2)

    # Differentiation matrix D_ref via barycentric weights.
    def lagrange_D(x):
        n = x.size
        diff = x[:, None] - x[None, :]
        offdiag = ~np.eye(n, dtype=bool)
        safe_diff = diff.copy()
        safe_diff[~offdiag] = 1.0
        bary = 1.0 / np.prod(safe_diff, axis=1)
        D = np.zeros((n, n), dtype=np.float64)
        np.divide(
            bary[None, :] / bary[:, None],
            diff,
            out=D,
            where=offdiag,
        )
        D[~offdiag] = -np.sum(D, axis=1)
        return D
    D_ref = lagrange_D(xi)
    # physical D: D_phys = (2/h) D_ref (chain rule)
    D = (2.0 / h) * D_ref

    # init u
    if u0 is None:
        def u0(x):
            return np.sin(2 * np.pi * x / L)
    u = u0(x_all).reshape(n_cells, n_local)

    dt = cfl * h / abs(c) / (2 * p + 1)
    n_steps = int(np.ceil(T / dt))
    dt = T / n_steps
    U = np.zeros((n_cells * n_local, n_steps + 1))
    U[:, 0] = u.reshape(-1)
    t = np.zeros(n_steps + 1)

    k = 0
    while k < n_steps:
        # upwind numerical flux at cell boundaries
        u_left = u[:, 0]
        u_right = u[:, -1]  # right face of each cell
        # upwind: if c > 0, flux = c * u_right_prev (left neighbor)
        if c > 0:
            flux_left = c * np.roll(u_right, 1)   # from left cell
            flux_right = c * u_right
        else:
            flux_left = c * u_left
            flux_right = c * np.roll(u_left, -1)
        # Interior contribution plus lumped-mass boundary terms.
        du = -c * (u @ D.T)
        du[:, 0] -= (flux_left - c * u_left) * (2.0 / h) / w[0]
        du[:, -1] += (flux_right - c * u_right) * (2.0 / h) / w[-1]
        u = u + dt * du
        U[:, k + 1] = u.reshape(-1)
        t[k + 1] = (k + 1) * dt
        k += 1
    return x_all, t, U


__all__ = ["solve_advection_1d_dg"]
