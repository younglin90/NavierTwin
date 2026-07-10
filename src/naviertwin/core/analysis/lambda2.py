"""Lambda-2 criterion — 소용돌이 식별.

Λ₂ = 2nd eigenvalue of (S² + Ω²), Λ₂ < 0 → vortex.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.lambda2 import lambda2_2d
    >>> u = np.zeros((32, 32)); v = np.zeros((32, 32))
    >>> L = lambda2_2d(u, v)
    >>> L.shape
    (32, 32)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.core.analysis.velocity_gradient import (
    _check_same_2d,
    _check_spacing,
    _grad_x,
    _grad_y,
)


def lambda2_2d(
    u: NDArray[np.float64], v: NDArray[np.float64],
    dx: float = 1.0, dy: float = 1.0,
) -> NDArray[np.float64]:
    """2D Λ₂ (두 번째 eigenvalue). 음수 → vortex."""
    if _kernels is None:
        return _lambda2_2d_numpy(u, v, dx, dy)
    return _kernels.lambda2_2d(u, v, dx, dy)


def _lambda2_2d_numpy(
    u: NDArray[np.float64], v: NDArray[np.float64],
    dx: float = 1.0, dy: float = 1.0,
) -> NDArray[np.float64]:
    """NumPy fallback matching the native 2D lambda2 kernel."""
    u_arr, v_arr = _check_same_2d(u, v)
    _check_spacing(dx, dy)

    du_dx = _grad_x(u_arr, dx)
    du_dy = _grad_y(u_arr, dy)
    dv_dx = _grad_x(v_arr, dx)
    dv_dy = _grad_y(v_arr, dy)

    s11 = du_dx
    s22 = dv_dy
    s12 = 0.5 * (du_dy + dv_dx)
    o12 = 0.5 * (dv_dx - du_dy)

    m11 = s11 * s11 + s12 * s12 - o12 * o12
    m22 = s22 * s22 + s12 * s12 - o12 * o12
    m12 = s11 * s12 + s12 * s22
    mid = 0.5 * (m11 + m22)
    rad = np.sqrt(0.25 * (m11 - m22) * (m11 - m22) + m12 * m12)
    return mid - rad


__all__ = ["lambda2_2d"]
