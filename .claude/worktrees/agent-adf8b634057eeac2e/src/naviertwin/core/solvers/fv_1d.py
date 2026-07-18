"""1D finite-volume scalar conservation law — Godunov upwind + Lax-Friedrichs.

∂u/∂t + ∂f(u)/∂x = 0.

Examples:
    >>> from naviertwin.core.solvers.fv_1d import solve_conservation_1d
    >>> import numpy as np
    >>> x, t, U = solve_conservation_1d(n_cells=50, T=0.2, flux=lambda u: u)
    >>> U.shape[0]
    50
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import HAS_NATIVE_KERNELS, _kernels


def solve_conservation_1d(
    n_cells: int = 100, L: float = 1.0, T: float = 0.2,
    flux: Callable[[NDArray], NDArray] | None = None,
    u0: Callable[[NDArray], NDArray] | None = None,
    *, scheme: str = "lxf", dt_factor: float = 0.4, c_max: float = 1.0,
) -> tuple[NDArray, NDArray, NDArray]:
    """periodic BC. Returns (x, t, U[n_cells, n_steps+1])."""
    native_linear = (
        getattr(_kernels, "conservation_1d_linear", None)
        if HAS_NATIVE_KERNELS
        else None
    )
    if flux is None and u0 is None and native_linear is not None:
        return native_linear(
            n_cells,
            L,
            T,
            scheme,
            dt_factor,
            c_max,
        )

    dx = L / n_cells
    x = np.linspace(0.5 * dx, L - 0.5 * dx, n_cells)
    u = u0(x) if u0 else np.sin(2 * np.pi * x / L)
    f = flux if flux else (lambda u: u)
    dt = dt_factor * dx / c_max
    n_steps = int(np.ceil(T / dt))
    dt = T / n_steps

    U = np.zeros((n_cells, n_steps + 1))
    U[:, 0] = u
    t = np.zeros(n_steps + 1)

    k = 0
    while k < n_steps:
        u_L = np.roll(u, 1)
        u_R = np.roll(u, -1)
        f_center = f(u)
        f_left = f(u_L)
        f_right = f(u_R)
        if scheme == "lxf":
            # Lax-Friedrichs numerical flux at i+1/2
            flux_right = 0.5 * (f_center + f_right) - 0.5 * (dx / dt) * (u_R - u)
            flux_left = 0.5 * (f_left + f_center) - 0.5 * (dx / dt) * (u - u_L)
        else:
            # upwind (c>0 가정)
            flux_right = f_center
            flux_left = f_left
        u = u - dt / dx * (flux_right - flux_left)
        U[:, k + 1] = u
        t[k + 1] = (k + 1) * dt
        k += 1
    return x, t, U


__all__ = ["solve_conservation_1d"]
