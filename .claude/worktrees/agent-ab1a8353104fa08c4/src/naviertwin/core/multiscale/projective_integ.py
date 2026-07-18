"""Equation-free projective integration — Kevrekidis style.

Inner micro burst → estimate slope → big projective step.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.multiscale.projective_integ import projective_step
    >>> def micro(u, dt): return u * np.exp(-dt)
    >>> u = np.array([1.0])
    >>> u2 = projective_step(u, micro, dt_micro=0.01, n_micro=5, dt_big=1.0)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def projective_step(
    u0: NDArray[np.float64],
    micro_step: Callable[[NDArray, float], NDArray],
    *,
    dt_micro: float,
    n_micro: int,
    dt_big: float,
) -> NDArray[np.float64]:
    """k inner steps, fit slope, then forward Euler with dt_big."""
    u = np.asarray(u0, dtype=np.float64).copy()
    history = [u.copy()]
    step = 0
    while step < n_micro:
        u = micro_step(u, dt_micro)
        history.append(u.copy())
        step += 1
    H = np.asarray(history)  # (n_micro+1, dim)
    slope = (H[-1] - H[-2]) / dt_micro
    return u + dt_big * slope


__all__ = ["projective_step"]
