"""1D FD 솔버 — heat / Burgers (ROM 검증 데이터 생성용).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solvers.fd_1d import solve_heat_1d
    >>> x, t, U = solve_heat_1d(nx=32, L=1.0, T=0.1, nu=0.01, u0=lambda x: np.sin(np.pi*x))
    >>> U.shape[0] > 0
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def solve_heat_1d(
    nx: int = 64, L: float = 1.0, T: float = 0.1,
    nu: float = 0.01, u0: Callable | None = None,
    *, dt_factor: float = 0.4,
) -> tuple[NDArray, NDArray, NDArray]:
    """1D 확산 방정식 u_t = ν u_xx (Dirichlet 0 경계).

    Returns:
        (x, t, U[nx, n_times]).
    """
    dx = L / (nx - 1)
    dt = dt_factor * dx ** 2 / (2 * nu)
    n_steps = int(np.ceil(T / dt))
    dt = T / n_steps
    x = np.linspace(0, L, nx)
    if u0 is None:
        u = np.sin(np.pi * x)
    else:
        u = u0(x)
    u[0] = u[-1] = 0.0
    U = np.zeros((nx, n_steps + 1), dtype=np.float64)
    U[:, 0] = u
    t = np.zeros(n_steps + 1)
    coef = nu * dt / dx ** 2
    for k in range(n_steps):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + coef * (u[2:] - 2 * u[1:-1] + u[:-2])
        u_new[0] = u_new[-1] = 0.0
        u = u_new
        U[:, k + 1] = u
        t[k + 1] = (k + 1) * dt
    return x, t, U


def solve_burgers_1d(
    nx: int = 128, L: float = 1.0, T: float = 0.2,
    nu: float = 0.01, u0: Callable | None = None,
    *, dt_factor: float = 0.4,
) -> tuple[NDArray, NDArray, NDArray]:
    """viscous Burgers u_t + u u_x = ν u_xx (Dirichlet 0)."""
    dx = L / (nx - 1)
    dt = dt_factor * min(dx / 2.0, dx ** 2 / (2 * nu))
    n_steps = int(np.ceil(T / dt))
    dt = T / n_steps
    x = np.linspace(0, L, nx)
    u = u0(x) if u0 is not None else np.sin(np.pi * x)
    u[0] = u[-1] = 0.0
    U = np.zeros((nx, n_steps + 1), dtype=np.float64)
    U[:, 0] = u
    t = np.zeros(n_steps + 1)
    for k in range(n_steps):
        # 중심차분 + upwind 혼합 (단순 central)
        du = (u[2:] - u[:-2]) / (2 * dx)
        d2u = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + dt * (-u[1:-1] * du + nu * d2u)
        u_new[0] = u_new[-1] = 0.0
        u = u_new
        U[:, k + 1] = u
        t[k + 1] = (k + 1) * dt
    return x, t, U


__all__ = ["solve_heat_1d", "solve_burgers_1d"]
