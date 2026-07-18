"""2D 비압축 Navier-Stokes — Chorin projection (교육용 콤팩트).

staggered 가 아닌 collocated + 단순 FD. 정확도보다는 ROM/해석 데이터 생성용.

Examples:
    >>> from naviertwin.core.solvers.ns_projection_2d import solve_cavity
    >>> u, v, p = solve_cavity(nx=16, ny=16, Re=100.0, n_steps=50)
    >>> u.shape
    (16, 16)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.solvers.pressure_poisson import poisson_2d_jacobi


def solve_cavity(
    nx: int = 32, ny: int = 32,
    Re: float = 100.0, n_steps: int = 200,
    *, Lx: float = 1.0, Ly: float = 1.0,
    U_lid: float = 1.0,
    dt: float | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Lid-driven cavity (top lid U=U_lid, others walls no-slip).

    Returns:
        (u, v, p) final fields, shape (nx, ny) each.
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    nu = 1.0 / Re
    if dt is None:
        dt_diff = 0.25 * min(dx ** 2, dy ** 2) / nu
        dt_conv = 0.25 * min(dx, dy) / max(abs(U_lid), 1e-6)
        dt = min(dt_diff, dt_conv)
    u = np.zeros((nx, ny), dtype=np.float64)
    v = np.zeros((nx, ny), dtype=np.float64)
    p = np.zeros((nx, ny), dtype=np.float64)

    def _apply_bc(u, v):
        # walls first, lid last so corners take lid value
        u[:, 0] = 0.0
        v[:, 0] = 0.0
        u[0, :] = 0.0
        v[0, :] = 0.0
        u[-1, :] = 0.0
        v[-1, :] = 0.0
        u[:, -1] = U_lid
        v[:, -1] = 0.0

    step = 0
    while step < n_steps:
        _apply_bc(u, v)

        # convection (upwind) + diffusion (central)
        # predictor stage
        du_x = (u[1:-1, 1:-1] - u[:-2, 1:-1]) / dx
        du_y = (u[1:-1, 1:-1] - u[1:-1, :-2]) / dy
        dv_x = (v[1:-1, 1:-1] - v[:-2, 1:-1]) / dx
        dv_y = (v[1:-1, 1:-1] - v[1:-1, :-2]) / dy
        lap_u = (
            (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx ** 2
            + (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy ** 2
        )
        lap_v = (
            (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx ** 2
            + (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dy ** 2
        )
        u_star = u.copy()
        v_star = v.copy()
        u_star[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (
            -u[1:-1, 1:-1] * du_x - v[1:-1, 1:-1] * du_y + nu * lap_u
        )
        v_star[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * (
            -u[1:-1, 1:-1] * dv_x - v[1:-1, 1:-1] * dv_y + nu * lap_v
        )
        # pressure Poisson: ∇²p = (1/dt) ∇·u*
        div_us = (
            (u_star[2:, 1:-1] - u_star[:-2, 1:-1]) / (2 * dx)
            + (v_star[1:-1, 2:] - v_star[1:-1, :-2]) / (2 * dy)
        )
        rhs = np.zeros_like(p)
        rhs[1:-1, 1:-1] = div_us / dt
        p, _ = poisson_2d_jacobi(rhs, dx, dy, max_iter=200, tol=1e-4)

        # corrector
        dp_x = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dx)
        dp_y = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dy)
        u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt * dp_x
        v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt * dp_y
        step += 1
    _apply_bc(u, v)
    return u, v, p


__all__ = ["solve_cavity"]
