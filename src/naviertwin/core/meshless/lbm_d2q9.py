"""Lattice-Boltzmann D2Q9 — BGK collision + streaming (single step).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.meshless.lbm_d2q9 import lbm_step
    >>> f = np.ones((9, 5, 5)) * (1.0 / 9.0)
    >>> f2 = lbm_step(f, omega=1.0)
    >>> f2.shape
    (9, 5, 5)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")

# D2Q9 velocities and weights
_E = np.array([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1],
])
_W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)


def equilibrium(rho: NDArray, u: NDArray) -> NDArray:
    """rho: (X,Y); u: (2,X,Y) → f_eq: (9,X,Y)."""
    return _kernels.lbm_equilibrium(
        np.asarray(rho, dtype=np.float64),
        np.asarray(u, dtype=np.float64),
    )


def lbm_step(f: NDArray, *, omega: float = 1.0) -> NDArray:
    """BGK collision + streaming (periodic)."""
    return _kernels.lbm_step(np.asarray(f, dtype=np.float64), float(omega))


__all__ = ["equilibrium", "lbm_step"]
