"""VOF (Volume of Fluid) 1D — color function advection (upwind).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.coupling.vof_1d import vof_step
    >>> alpha = np.array([1, 1, 0.5, 0, 0], dtype=float)
    >>> u = np.ones(5)
    >>> a2 = vof_step(alpha, u, dt=0.1, dx=0.1)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def vof_step(
    alpha: NDArray[np.float64],
    u: NDArray[np.float64],
    *,
    dt: float, dx: float,
) -> NDArray[np.float64]:
    """upwind advection of color function (clipped to [0, 1])."""
    return _kernels.vof_step_1d(
        np.asarray(alpha, dtype=np.float64),
        np.asarray(u, dtype=np.float64),
        float(dt),
        float(dx),
    )


__all__ = ["vof_step"]
