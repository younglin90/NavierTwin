"""2D 이류-확산 솔버 — upwind + 중심차분 (유한차분/유한체적 혼합).

∂c/∂t + U·∇c = D ∇²c, 정상 U=(u0, v0).

Examples:
    >>> from naviertwin.core.solvers.conv_diff_2d import solve_conv_diff_2d
    >>> x, y, t, C = solve_conv_diff_2d(nx=32, ny=32, T=0.05)
    >>> C.shape[-1] > 0
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

try:
    from naviertwin._native import _kernels
except Exception:  # pragma: no cover - optional extension
    _kernels = None


def solve_conv_diff_2d(
    nx: int = 64,
    ny: int = 64,
    Lx: float = 1.0,
    Ly: float = 1.0,
    T: float = 0.1,
    u0: float = 1.0,
    v0: float = 0.5,
    D: float = 0.01,
    c0: Callable | None = None,
    *,
    dt_factor: float = 0.3,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    # CFL conditions
    dt_conv = 1.0 / (abs(u0) / dx + abs(v0) / dy + 1e-30)
    dt_diff = 1.0 / (2 * D * (1 / dx ** 2 + 1 / dy ** 2) + 1e-30)
    dt = dt_factor * min(dt_conv, dt_diff)
    n_steps = int(np.ceil(T / dt))
    dt = T / n_steps

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    if c0 is None:
        # 가우시안 범프
        c = np.exp(-50 * ((X - 0.3) ** 2 + (Y - 0.3) ** 2))
    else:
        c = c0(X, Y)

    if _kernels is not None:
        t, c_hist = _kernels.conv_diff_2d_evolve(c, n_steps, u0, v0, D, dx, dy, dt)
        return x, y, t, c_hist

    C = np.zeros((nx, ny, n_steps + 1), dtype=np.float64)
    C[:, :, 0] = c
    t = np.zeros(n_steps + 1)

    k = 0
    while k < n_steps:
        # 1차 upwind convection
        if u0 >= 0:
            dcdx = (c[1:-1, 1:-1] - c[:-2, 1:-1]) / dx
        else:
            dcdx = (c[2:, 1:-1] - c[1:-1, 1:-1]) / dx
        if v0 >= 0:
            dcdy = (c[1:-1, 1:-1] - c[1:-1, :-2]) / dy
        else:
            dcdy = (c[1:-1, 2:] - c[1:-1, 1:-1]) / dy
        # 2차 중심차분 diffusion
        d2c = (
            (c[2:, 1:-1] - 2 * c[1:-1, 1:-1] + c[:-2, 1:-1]) / dx**2
            + (c[1:-1, 2:] - 2 * c[1:-1, 1:-1] + c[1:-1, :-2]) / dy**2
        )
        c_new = c.copy()
        c_new[1:-1, 1:-1] = c[1:-1, 1:-1] + dt * (
            -u0 * dcdx - v0 * dcdy + D * d2c
        )
        c = c_new
        C[:, :, k + 1] = c
        t[k + 1] = (k + 1) * dt
        k += 1
    return x, y, t, C


__all__ = ["solve_conv_diff_2d"]
