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


def lambda2_2d(
    u: NDArray[np.float64], v: NDArray[np.float64],
    dx: float = 1.0, dy: float = 1.0,
) -> NDArray[np.float64]:
    """2D Λ₂ (두 번째 eigenvalue). 음수 → vortex."""
    if _kernels is None:
        raise ImportError("lambda2_2d requires the NavierTwin C++ native kernels")
    return _kernels.lambda2_2d(u, v, dx, dy)


__all__ = ["lambda2_2d"]
