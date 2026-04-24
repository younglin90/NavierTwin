"""SSP-RK3 — Shu-Osher 1988 strong-stability-preserving Runge-Kutta 3rd-order.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solvers.ssp_rk3 import ssp_rk3_step
    >>> def rhs(u): return -u
    >>> u = np.array([1.0])
    >>> u2 = ssp_rk3_step(u, rhs, dt=0.1)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def ssp_rk3_step(
    u: NDArray[np.float64],
    rhs: Callable[[NDArray], NDArray],
    *,
    dt: float,
) -> NDArray[np.float64]:
    """3-stage SSP-RK3."""
    u = np.asarray(u, dtype=np.float64)
    u1 = u + dt * rhs(u)
    u2 = 0.75 * u + 0.25 * (u1 + dt * rhs(u1))
    u3 = (1 / 3) * u + (2 / 3) * (u2 + dt * rhs(u2))
    return u3


__all__ = ["ssp_rk3_step"]
