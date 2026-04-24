"""Okubo-Weiss criterion (2D) — W = S² - ω², W < 0 → vortex.

S² = strain rate squared = s_n² + s_s²,  ω² = vorticity squared.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.okubo_weiss import okubo_weiss
    >>> grad = np.zeros((1, 2, 2))
    >>> grad[0] = [[0, -1], [1, 0]]
    >>> okubo_weiss(grad)[0]
    np.float64(-4.0)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def okubo_weiss(grad_u_2d: NDArray[np.float64]) -> NDArray[np.float64]:
    """grad shape (..., 2, 2) (∂u/∂x ∂u/∂y / ∂v/∂x ∂v/∂y) → W = S² - ω²."""
    g = np.asarray(grad_u_2d, dtype=np.float64)
    du_dx = g[..., 0, 0]
    du_dy = g[..., 0, 1]
    dv_dx = g[..., 1, 0]
    dv_dy = g[..., 1, 1]
    s_n = du_dx - dv_dy        # normal strain
    s_s = du_dy + dv_dx        # shear strain
    omega = dv_dx - du_dy      # vorticity (scalar in 2D)
    return s_n * s_n + s_s * s_s - omega * omega


__all__ = ["okubo_weiss"]
