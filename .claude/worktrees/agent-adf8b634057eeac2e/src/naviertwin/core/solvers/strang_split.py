"""Strang splitting — 2nd-order operator splitting.

u(t+dt) = exp(A·dt/2) exp(B·dt) exp(A·dt/2) u.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solvers.strang_split import strang_step
    >>> def A(u, dt): return u * np.exp(-0.5 * dt)
    >>> def B(u, dt): return u + dt
    >>> u = np.array([1.0])
    >>> u2 = strang_step(u, A, B, dt=0.1)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def strang_step(
    u: NDArray[np.float64],
    op_A: Callable[[NDArray, float], NDArray],
    op_B: Callable[[NDArray, float], NDArray],
    *,
    dt: float,
) -> NDArray[np.float64]:
    """exp(A·dt/2) ∘ exp(B·dt) ∘ exp(A·dt/2)."""
    u = np.asarray(u, dtype=np.float64).copy()
    u = op_A(u, dt / 2.0)
    u = op_B(u, dt)
    u = op_A(u, dt / 2.0)
    return u


__all__ = ["strang_step"]
