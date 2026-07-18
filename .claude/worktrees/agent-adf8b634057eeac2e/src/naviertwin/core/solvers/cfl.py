"""CFL 조건 기반 stable timestep 계산.

convective: dt ≤ CFL · dx / |u|
diffusive:  dt ≤ CFL · dx²/(2ν)

Examples:
    >>> from naviertwin.core.solvers.cfl import cfl_convective, cfl_diffusive
    >>> cfl_convective(dx=0.01, u_max=2.0, cfl=0.8)
    0.004
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cfl_convective(dx: float, u_max: float, cfl: float = 0.9) -> float:
    if u_max <= 0:
        return float("inf")
    return cfl * dx / u_max


def cfl_diffusive(dx: float, nu: float, cfl: float = 0.5, dim: int = 1) -> float:
    if nu <= 0:
        return float("inf")
    return cfl * dx ** 2 / (2 * dim * nu)


def cfl_combined(
    dx: float, u_max: float, nu: float,
    *, cfl_conv: float = 0.9, cfl_diff: float = 0.5, dim: int = 1,
) -> float:
    return min(
        cfl_convective(dx, u_max, cfl_conv),
        cfl_diffusive(dx, nu, cfl_diff, dim),
    )


def cfl_number(dt: float, dx: float, u_max: float) -> float:
    if dx <= 0:
        return float("inf")
    return abs(u_max) * dt / dx


def cfl_field(
    dt: float, dx: float, dy: float, u: NDArray[np.float64], v: NDArray[np.float64],
) -> float:
    """2D 최대 국소 CFL."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return float(np.max(np.abs(u) * dt / dx + np.abs(v) * dt / dy))


__all__ = [
    "cfl_convective", "cfl_diffusive", "cfl_combined",
    "cfl_number", "cfl_field",
]
