"""1D FVM 이류 방정식 — upwind + MUSCL-Hancock.

    ∂u/∂t + c ∂u/∂x = 0,  주기 경계.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solver_interfaces.fvm_advection import (
    ...     fvm_upwind_1d, minmod,
    ... )
    >>> u0 = np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False))
    >>> t, U = fvm_upwind_1d(u0, c=1.0, L=2 * np.pi, T=0.5, cfl=0.5)
    >>> U.shape[1] == 64
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import HAS_NATIVE_KERNELS, _kernels
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def minmod(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Minmod slope limiter."""
    sa = np.sign(a)
    sb = np.sign(b)
    return 0.5 * (sa + sb) * np.minimum(np.abs(a), np.abs(b))


def fvm_upwind_1d(
    u0: NDArray[np.float64],
    c: float = 1.0,
    L: float = 2 * np.pi,
    T: float = 1.0,
    cfl: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """1D 주기 경계 upwind FVM.

    Returns:
        (times, U[n_steps, N]).
    """
    native_upwind = (
        getattr(_kernels, "fvm_upwind_1d", None)
        if HAS_NATIVE_KERNELS
        else None
    )
    if native_upwind is not None:
        return native_upwind(u0, c, L, T, cfl)

    u0 = np.asarray(u0, dtype=np.float64)
    N = u0.size
    dx = L / N
    dt = cfl * dx / abs(c)
    n_steps = int(T / dt) + 1
    times = np.linspace(dt, T, n_steps)
    U = np.zeros((n_steps, N))
    u = u0.copy()

    k = 0
    while k < n_steps:
        if c > 0:
            flux = u
            u = u - cfl * (flux - np.roll(flux, 1))
        else:
            flux = u
            u = u - cfl * (np.roll(flux, -1) - flux)
        U[k] = u
        k += 1
    return times, U


def fvm_musclhancock_1d(
    u0: NDArray[np.float64],
    c: float = 1.0,
    L: float = 2 * np.pi,
    T: float = 1.0,
    cfl: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """2차 정확 MUSCL-Hancock with minmod limiter."""
    native_muscl = (
        getattr(_kernels, "fvm_musclhancock_1d", None)
        if HAS_NATIVE_KERNELS
        else None
    )
    if native_muscl is not None:
        return native_muscl(u0, c, L, T, cfl)

    u0 = np.asarray(u0, dtype=np.float64)
    N = u0.size
    dx = L / N
    dt = cfl * dx / abs(c)
    n_steps = int(T / dt) + 1
    times = np.linspace(dt, T, n_steps)
    U = np.zeros((n_steps, N))
    u = u0.copy()

    k = 0
    while k < n_steps:
        # slopes
        dup = np.roll(u, -1) - u
        dum = u - np.roll(u, 1)
        slope = minmod(dum, dup)

        # left/right at interface
        u_L = u + 0.5 * slope
        u_R = np.roll(u - 0.5 * slope, -1)

        if c > 0:
            flux = c * u_L
        else:
            flux = c * u_R
        u = u - dt / dx * (flux - np.roll(flux, 1))
        U[k] = u
        k += 1
    return times, U


def total_mass(u: NDArray[np.float64], dx: float) -> float:
    """Σ u_i · dx — 주기 경계 보존량."""
    return float(np.sum(u) * dx)


__all__ = [
    "minmod",
    "fvm_upwind_1d",
    "fvm_musclhancock_1d",
    "total_mass",
]
