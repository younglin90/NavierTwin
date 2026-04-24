"""Genetic Algorithm — real-coded GA (SBX-like crossover + Gaussian mutation).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.genetic import ga
    >>> x, f = ga(lambda v: float(v @ v), bounds=[(-5, 5), (-5, 5)],
    ...           n_pop=30, n_gen=50, seed=0)
    >>> np.linalg.norm(x) < 0.5
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def ga(
    objective: Callable[[NDArray[np.float64]], float],
    bounds: list[tuple[float, float]],
    *, n_pop: int = 30, n_gen: int = 100,
    mut_rate: float = 0.1, mut_sigma: float = 0.1,
    elitism: int = 2, tourn_k: int = 3,
    seed: int | None = 0,
) -> tuple[NDArray[np.float64], float]:
    rng = np.random.default_rng(seed)
    d = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    pop = rng.uniform(lo, hi, size=(n_pop, d))
    fit = np.array([float(objective(p)) for p in pop])
    best_x = pop[np.argmin(fit)].copy()
    best_f = float(fit.min())

    for _ in range(n_gen):
        new_pop = []
        # elitism
        elite_idx = np.argsort(fit)[:elitism]
        for e in elite_idx:
            new_pop.append(pop[e].copy())

        while len(new_pop) < n_pop:
            # tournament selection
            def pick():
                cand = rng.integers(0, n_pop, tourn_k)
                return pop[cand[np.argmin(fit[cand])]]

            p1 = pick()
            p2 = pick()
            # uniform crossover + small Gaussian blend
            alpha = rng.uniform(-0.1, 1.1, size=d)
            child = alpha * p1 + (1 - alpha) * p2
            # mutation
            if rng.random() < mut_rate:
                child = child + rng.normal(0, mut_sigma * (hi - lo), size=d)
            child = np.clip(child, lo, hi)
            new_pop.append(child)

        pop = np.stack(new_pop[:n_pop], axis=0)
        fit = np.array([float(objective(p)) for p in pop])
        if fit.min() < best_f:
            best_f = float(fit.min())
            best_x = pop[np.argmin(fit)].copy()
    return best_x, best_f


__all__ = ["ga"]
