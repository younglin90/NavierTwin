"""Geometric Multigrid V-cycle — 2D Poisson on uniform grid.

Dirichlet 0 경계. full weighting restriction + bilinear prolongation.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.linalg.multigrid import v_cycle_poisson
    >>> n = 33
    >>> h = 1.0 / (n - 1)
    >>> x = np.linspace(0, 1, n)
    >>> X, Y = np.meshgrid(x, x, indexing='ij')
    >>> f = -2 * (np.pi ** 2) * np.sin(np.pi * X) * np.sin(np.pi * Y)
    >>> u = np.zeros_like(f)
    >>> u = v_cycle_poisson(u, f, h, n_pre=2, n_post=2, levels=3)
    >>> u.shape
    (33, 33)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _gauss_seidel_sweep(u: NDArray, f: NDArray, h: float, n: int = 1) -> NDArray:
    """-Δu = f, Dirichlet 0. GS 업데이트: u = 0.25 (Σ neighbors + h² f)."""
    u = u.copy()
    h2 = h * h
    for _ in range(n):
        u[1:-1, 1:-1] = 0.25 * (
            u[2:, 1:-1] + u[:-2, 1:-1]
            + u[1:-1, 2:] + u[1:-1, :-2]
            + h2 * f[1:-1, 1:-1]
        )
    return u


def _residual(u: NDArray, f: NDArray, h: float) -> NDArray:
    """r = f - (-Δu)_h, (-Δu)_h = (4u - neighbors)/h²."""
    r = np.zeros_like(u)
    lap = (
        4 * u[1:-1, 1:-1]
        - u[2:, 1:-1] - u[:-2, 1:-1]
        - u[1:-1, 2:] - u[1:-1, :-2]
    ) / (h * h)
    r[1:-1, 1:-1] = f[1:-1, 1:-1] - lap
    return r


def _restrict(r: NDArray) -> NDArray:
    """full weighting restriction: (N,N) → ((N+1)/2, (N+1)/2)."""
    n = r.shape[0]
    m = (n - 1) // 2 + 1
    out = np.zeros((m, m))
    # 내부만 가중 평균 (경계는 0)
    for i in range(1, m - 1):
        for j in range(1, m - 1):
            ii, jj = 2 * i, 2 * j
            out[i, j] = (
                4 * r[ii, jj]
                + 2 * (r[ii + 1, jj] + r[ii - 1, jj] + r[ii, jj + 1] + r[ii, jj - 1])
                + (r[ii + 1, jj + 1] + r[ii + 1, jj - 1]
                   + r[ii - 1, jj + 1] + r[ii - 1, jj - 1])
            ) / 16.0
    return out


def _prolong(e: NDArray, n_fine: int) -> NDArray:
    """bilinear prolongation: coarse → fine."""
    out = np.zeros((n_fine, n_fine))
    # copy coarse to fine grid at even indices
    out[::2, ::2] = e
    # interpolate horizontally
    out[::2, 1::2] = 0.5 * (out[::2, :-1:2] + out[::2, 2::2])
    # interpolate vertically
    out[1::2, :] = 0.5 * (out[:-1:2, :] + out[2::2, :])
    return out


def v_cycle_poisson(
    u: NDArray[np.float64], f: NDArray[np.float64], h: float,
    *, n_pre: int = 2, n_post: int = 2, levels: int = 3,
) -> NDArray[np.float64]:
    """2D Poisson -∇²u = f (Dirichlet 0) V-cycle 1 회."""
    if levels <= 1 or u.shape[0] <= 3:
        # 최하층: 다수 sweep 로 근사 해
        return _gauss_seidel_sweep(u, f, h, n=50)
    u = _gauss_seidel_sweep(u, f, h, n=n_pre)
    r = _residual(u, f, h)
    rc = _restrict(r)
    ec = np.zeros_like(rc)
    ec = v_cycle_poisson(ec, rc, 2 * h, n_pre=n_pre, n_post=n_post, levels=levels - 1)
    e = _prolong(ec, u.shape[0])
    u = u + e
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
    return _gauss_seidel_sweep(u, f, h, n=n_post)


def solve_poisson_multigrid(
    f: NDArray[np.float64], h: float,
    *, max_cycles: int = 20, tol: float = 1e-8,
    n_pre: int = 2, n_post: int = 2, levels: int = 4,
) -> tuple[NDArray[np.float64], dict]:
    u = np.zeros_like(f)
    for i in range(max_cycles):
        u = v_cycle_poisson(u, f, h, n_pre=n_pre, n_post=n_post, levels=levels)
        r = _residual(u, f, h)
        res = float(np.linalg.norm(r))
        if res < tol:
            return u, {"cycles": i + 1, "residual": res, "converged": True}
    return u, {"cycles": max_cycles, "residual": res, "converged": False}


__all__ = ["v_cycle_poisson", "solve_poisson_multigrid"]
