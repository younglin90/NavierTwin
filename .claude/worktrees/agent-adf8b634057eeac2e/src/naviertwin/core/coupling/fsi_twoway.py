"""FSI 2-way — fixed-point iteration with Aitken relaxation.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.coupling.fsi_twoway import aitken_relax
    >>> r0 = np.array([1.0]); r1 = np.array([0.5])
    >>> w = aitken_relax(0.5, r0, r1)
    >>> 0 < w < 2
    True
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required by FSI coupling")


def aitken_relax(
    omega_prev: float,
    r_prev: NDArray[np.float64],
    r_curr: NDArray[np.float64],
) -> float:
    """Aitken: ω_new = -ω_prev * (r_prev · (r_curr - r_prev)) / ‖r_curr - r_prev‖²."""
    return float(
        _kernels.aitken_relax(
            float(omega_prev),
            np.asarray(r_prev, dtype=np.float64),
            np.asarray(r_curr, dtype=np.float64),
        ),
    )


def fsi_2way(
    fluid_solve: Callable[[NDArray], NDArray],
    solid_solve: Callable[[NDArray], NDArray],
    x0: NDArray[np.float64],
    *,
    n_iter: int = 30,
    tol: float = 1e-6,
    omega0: float = 0.5,
) -> NDArray[np.float64]:
    """fluid → solid (변위) → fluid 반복 with Aitken."""
    x = np.asarray(x0, dtype=np.float64).copy()
    omega = omega0
    r_prev = None
    it = 0
    while it < n_iter:
        load = fluid_solve(x)
        x_tilde = solid_solve(load)
        r = x_tilde - x
        if _kernels.vector_l2_norm(np.asarray(r, dtype=np.float64)) < tol:
            return x_tilde
        if r_prev is not None:
            omega = aitken_relax(omega, r_prev, r)
            omega = float(np.clip(omega, 0.05, 1.0))
        x = x + omega * r
        r_prev = r
        it += 1
    return x


__all__ = ["aitken_relax", "fsi_2way"]
