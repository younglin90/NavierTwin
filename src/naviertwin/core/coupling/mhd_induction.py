"""MHD induction equation 1D — ∂B/∂t = ∂/∂x(u B) + η ∂²B/∂x².

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.coupling.mhd_induction import induction_step
    >>> B = np.zeros(20); B[10] = 1.0
    >>> u = np.ones(20)
    >>> B2 = induction_step(B, u, dt=0.01, dx=0.1, eta=0.01)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def induction_step(
    B: NDArray[np.float64],
    u: NDArray[np.float64],
    *,
    dt: float, dx: float, eta: float = 0.0,
) -> NDArray[np.float64]:
    """upwind advection + central diffusion (zero-flux BC)."""
    B = np.asarray(B, dtype=np.float64).copy()
    u = np.asarray(u, dtype=np.float64)
    flux = u * B
    # central derivative (zero-flux at edges via mirror)
    df = np.zeros_like(B)
    df[1:-1] = (flux[2:] - flux[:-2]) / (2 * dx)
    lap = np.zeros_like(B)
    lap[1:-1] = (B[2:] - 2 * B[1:-1] + B[:-2]) / (dx * dx)
    return B + dt * (df + eta * lap)


__all__ = ["induction_step"]
