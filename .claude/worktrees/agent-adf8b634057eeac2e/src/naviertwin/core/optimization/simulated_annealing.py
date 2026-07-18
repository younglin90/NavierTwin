"""Simulated Annealing — 온도 감소형 stochastic 최적화.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.simulated_annealing import sa
    >>> x, f = sa(lambda v: float(v @ v), x0=np.array([5.0, -3.0]),
    ...            n_iter=2000, step=0.5, T0=1.0, seed=0)
    >>> np.linalg.norm(x) < 1.0
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def sa(
    objective: Callable[[NDArray[np.float64]], float],
    x0: NDArray[np.float64],
    *, n_iter: int = 1000, step: float = 0.5,
    T0: float = 1.0, cooling: float = 0.995,
    seed: int | None = 0,
) -> tuple[NDArray[np.float64], float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    f = float(objective(x))
    best_x = x.copy()
    best_f = f
    T = float(T0)
    it = 0
    while it < n_iter:
        prop = x + rng.normal(0, step, size=x.size)
        fp = float(objective(prop))
        if fp < f or rng.random() < np.exp(-(fp - f) / max(T, 1e-30)):
            x = prop
            f = fp
            if f < best_f:
                best_x = x.copy()
                best_f = f
        T *= cooling
        it += 1
    return best_x, best_f


__all__ = ["sa"]
