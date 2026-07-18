"""Cahn-Hilliard 1D — c_t = M ∇² (f'(c) - ε² ∇² c).  f(c) = c²(1-c)².

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.coupling.cahn_hilliard import ch_step
    >>> rng = np.random.default_rng(0)
    >>> c = 0.5 + 0.05 * rng.standard_normal(40)
    >>> c2 = ch_step(c, dt=1e-4, dx=0.05)
    >>> c2.shape
    (40,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _lap(c, dx):
    return (np.roll(c, -1) - 2 * c + np.roll(c, 1)) / (dx * dx)


def ch_step(
    c: NDArray[np.float64], *,
    dt: float = 1e-4, dx: float = 0.1,
    M: float = 1.0, eps: float = 0.05,
) -> NDArray[np.float64]:
    """1 explicit Euler step (periodic BC). f'(c) = 2 c (1-c)(1 - 2c)."""
    c = np.asarray(c, dtype=np.float64).copy()
    df = 2.0 * c * (1 - c) * (1 - 2 * c)
    mu = df - eps * eps * _lap(c, dx)
    return c + dt * M * _lap(mu, dx)


__all__ = ["ch_step"]
