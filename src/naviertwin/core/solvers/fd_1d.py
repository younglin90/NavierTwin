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

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


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
        u = np.asarray(u0(x), dtype=np.float64)
    u[0] = u[-1] = 0.0
    coef = nu * dt / dx ** 2
    t, U = _kernels.fd_heat_1d_evolve(u, n_steps, coef, dt)
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
    u = np.asarray(u0(x) if u0 is not None else np.sin(np.pi * x), dtype=np.float64)
    u[0] = u[-1] = 0.0
    t, U = _kernels.fd_burgers_1d_evolve(u, n_steps, dt, dx, nu)
    return x, t, U


__all__ = ["solve_heat_1d", "solve_burgers_1d"]
