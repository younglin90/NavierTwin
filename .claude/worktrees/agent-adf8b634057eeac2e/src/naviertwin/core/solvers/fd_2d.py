"""2D FD 솔버 — heat equation (FTCS).

Examples:
    >>> from naviertwin.core.solvers.fd_2d import solve_heat_2d
    >>> x, y, t, U = solve_heat_2d(nx=32, ny=32, T=0.05, nu=0.01)
    >>> U.shape[:2]
    (32, 32)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required by 2D finite-difference solvers")


def solve_heat_2d(
    nx: int = 64, ny: int = 64,
    Lx: float = 1.0, Ly: float = 1.0, T: float = 0.1,
    nu: float = 0.01,
    u0: Callable | None = None,
    *, dt_factor: float = 0.2,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """2D 열방정식 u_t = ν(u_xx + u_yy), Dirichlet 0.

    Returns:
        (x, y, t, U[nx, ny, n_times]).
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dt = dt_factor / (nu * (1 / dx ** 2 + 1 / dy ** 2))
    n_steps = int(np.ceil(T / dt))
    dt = T / n_steps
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    if u0 is None:
        u = np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
    else:
        u = u0(X, Y)
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0

    cx = nu * dt / dx ** 2
    cy = nu * dt / dy ** 2
    t, U = _kernels.fd_heat_2d_evolve(
        np.asarray(u, dtype=np.float64),
        int(n_steps),
        float(cx),
        float(cy),
        float(dt),
    )
    return x, y, t, U


__all__ = ["solve_heat_2d"]
