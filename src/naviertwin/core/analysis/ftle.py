"""FTLE (Finite-Time Lyapunov Exponent) — 2D Lagrangian coherent structures 근사.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.ftle import compute_ftle_2d
    >>> def vf(p): return np.array([-p[1], p[0]])  # rotation
    >>> ftle = compute_ftle_2d(vf, x=np.linspace(-1,1,10), y=np.linspace(-1,1,10), T=1.0, dt=0.01)
    >>> ftle.shape
    (10, 10)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels


def _advect(
    vf: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    pts: NDArray[np.float64], T: float, dt: float,
) -> NDArray[np.float64]:
    """RK4 로 pts (N, 2) 를 [0, T] 에 대해 advection."""
    n = int(np.ceil(T / dt))
    dt = T / n
    P = pts.copy()
    step = 0
    while step < n:
        k1 = _eval_vf(vf, P)
        k2 = _eval_vf(vf, P + 0.5 * dt * k1)
        k3 = _eval_vf(vf, P + 0.5 * dt * k2)
        k4 = _eval_vf(vf, P + dt * k3)
        P = P + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        step += 1
    return P


def _eval_vf(
    vf: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    pts: NDArray[np.float64],
) -> NDArray[np.float64]:
    out = np.empty_like(pts)
    idx = 0
    while idx < pts.shape[0]:
        out[idx] = vf(pts[idx])
        idx += 1
    return out


def compute_ftle_2d(
    vf: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: NDArray[np.float64], y: NDArray[np.float64],
    T: float = 1.0, dt: float = 0.01,
    *, eps: float = 1e-4,
) -> NDArray[np.float64]:
    """grid 각 점에서 FTLE = (1/(2T)) log λ_max(Cauchy-Green)."""
    if _kernels is None:
        raise ImportError("naviertwin._native._kernels is required by compute_ftle_2d")
    nx = x.size
    ny = y.size
    X, Y = np.meshgrid(x, y, indexing="xy")
    advected = np.empty((ny, nx, 4, 2), dtype=np.float64)
    i = 0
    while i < ny:
        j = 0
        while j < nx:
            p = np.array([X[i, j], Y[i, j]])
            stencil = np.array([
                p + [eps, 0],
                p - [eps, 0],
                p + [0, eps],
                p - [0, eps],
            ])
            advected[i, j] = _advect(vf, stencil, T, dt)
            j += 1
        i += 1
    return _kernels.ftle_from_advected_stencils(advected, T, eps)


__all__ = ["compute_ftle_2d"]
