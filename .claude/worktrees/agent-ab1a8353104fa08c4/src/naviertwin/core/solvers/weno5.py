"""WENO5 reconstruction — Jiang & Shu 1996, scalar 1D.

Returns u_{i+1/2}^L (left-biased) from stencil u_{i-2..i+2}.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solvers.weno5 import weno5_recon_left
    >>> u = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    >>> u_face = weno5_recon_left(u)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_EPS = 1e-6
_D = (0.1, 0.6, 0.3)  # ideal weights


def weno5_recon_left(u: NDArray[np.float64]) -> float:
    """5-point stencil → u_{i+1/2}^L (i = 2)."""
    u = np.asarray(u, dtype=np.float64)
    p0 = (1 / 3) * u[0] - (7 / 6) * u[1] + (11 / 6) * u[2]
    p1 = -(1 / 6) * u[1] + (5 / 6) * u[2] + (1 / 3) * u[3]
    p2 = (1 / 3) * u[2] + (5 / 6) * u[3] - (1 / 6) * u[4]
    beta0 = (13 / 12) * (u[0] - 2 * u[1] + u[2]) ** 2 + 0.25 * (u[0] - 4 * u[1] + 3 * u[2]) ** 2
    beta1 = (13 / 12) * (u[1] - 2 * u[2] + u[3]) ** 2 + 0.25 * (u[1] - u[3]) ** 2
    beta2 = (13 / 12) * (u[2] - 2 * u[3] + u[4]) ** 2 + 0.25 * (3 * u[2] - 4 * u[3] + u[4]) ** 2
    a0 = _D[0] / (_EPS + beta0) ** 2
    a1 = _D[1] / (_EPS + beta1) ** 2
    a2 = _D[2] / (_EPS + beta2) ** 2
    s = a0 + a1 + a2
    return float((a0 * p0 + a1 * p1 + a2 * p2) / s)


__all__ = ["weno5_recon_left"]
