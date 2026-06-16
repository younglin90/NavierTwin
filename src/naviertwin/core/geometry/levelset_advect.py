"""Level-set advection — φ_t + u·∇φ = 0, upwind 1D.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.geometry.levelset_advect import advect_step
    >>> phi = np.linspace(-1, 1, 11)
    >>> u = np.ones(11)
    >>> phi2 = advect_step(phi, u, dt=0.05, dx=0.2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def advect_step(
    phi: NDArray[np.float64],
    u: NDArray[np.float64],
    *,
    dt: float, dx: float,
) -> NDArray[np.float64]:
    return _kernels.levelset_advect_step_1d(
        np.asarray(phi, dtype=np.float64),
        np.asarray(u, dtype=np.float64),
        float(dt),
        float(dx),
    )


__all__ = ["advect_step"]
