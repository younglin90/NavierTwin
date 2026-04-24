"""Particle Swarm Optimization — 개구간 목적함수 최적화.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.pso import pso
    >>> x, f = pso(lambda x: float(x @ x), bounds=[(-5, 5), (-5, 5)],
    ...            n_particles=20, n_iter=50, seed=0)
    >>> np.linalg.norm(x) < 1e-2
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def pso(
    objective: Callable[[NDArray[np.float64]], float],
    bounds: list[tuple[float, float]],
    *, n_particles: int = 30, n_iter: int = 100,
    w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
    seed: int | None = 0,
) -> tuple[NDArray[np.float64], float]:
    rng = np.random.default_rng(seed)
    d = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x = rng.uniform(lo, hi, size=(n_particles, d))
    v = rng.uniform(-(hi - lo), hi - lo, size=(n_particles, d)) * 0.1
    f = np.array([float(objective(xi)) for xi in x])
    pbest = x.copy()
    pbest_f = f.copy()
    g_idx = int(np.argmin(pbest_f))
    gbest = pbest[g_idx].copy()
    gbest_f = float(pbest_f[g_idx])

    for _ in range(n_iter):
        r1 = rng.random((n_particles, d))
        r2 = rng.random((n_particles, d))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        x = np.clip(x + v, lo, hi)
        f = np.array([float(objective(xi)) for xi in x])
        better = f < pbest_f
        pbest[better] = x[better]
        pbest_f[better] = f[better]
        g_idx = int(np.argmin(pbest_f))
        if pbest_f[g_idx] < gbest_f:
            gbest = pbest[g_idx].copy()
            gbest_f = float(pbest_f[g_idx])
    return gbest, gbest_f


__all__ = ["pso"]
