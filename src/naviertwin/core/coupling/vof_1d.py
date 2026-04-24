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


def vof_step(
    alpha: NDArray[np.float64],
    u: NDArray[np.float64],
    *,
    dt: float, dx: float,
) -> NDArray[np.float64]:
    """upwind advection of color function (clipped to [0, 1])."""
    a = np.asarray(alpha, dtype=np.float64).copy()
    u = np.asarray(u, dtype=np.float64)
    a_new = a.copy()
    n = len(a)
    for i in range(1, n - 1):
        if u[i] >= 0:
            a_new[i] = a[i] - dt / dx * u[i] * (a[i] - a[i - 1])
        else:
            a_new[i] = a[i] - dt / dx * u[i] * (a[i + 1] - a[i])
    return np.clip(a_new, 0.0, 1.0)


__all__ = ["vof_step"]
