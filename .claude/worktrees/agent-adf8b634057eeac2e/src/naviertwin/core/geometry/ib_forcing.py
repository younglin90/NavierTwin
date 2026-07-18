"""Immersed Boundary forcing — direct forcing (Mohd-Yusof 1997 풍).

f = (u_target - u) / dt  on IB points.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.geometry.ib_forcing import direct_forcing
    >>> u = np.ones(5); mask = np.array([0, 0, 1, 0, 0], dtype=bool)
    >>> f = direct_forcing(u, np.zeros(5), mask, dt=0.1)
    >>> f[2] == -10.0
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def direct_forcing(
    u: NDArray[np.float64],
    u_target: NDArray[np.float64],
    mask: NDArray[np.bool_],
    *,
    dt: float = 1.0,
) -> NDArray[np.float64]:
    """f = mask * (u_target - u) / dt."""
    u = np.asarray(u)
    u_t = np.asarray(u_target)
    return np.asarray(mask, dtype=float) * (u_t - u) / dt


__all__ = ["direct_forcing"]
