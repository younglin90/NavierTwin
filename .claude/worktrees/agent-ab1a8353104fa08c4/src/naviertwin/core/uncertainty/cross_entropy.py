"""Cross-entropy method — rare event 추정 및 최적화.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.uncertainty.cross_entropy import cem_optimize
    >>> # 목적: x^T x 최소화 (최대화 반대)
    >>> x, f = cem_optimize(lambda v: -float(v @ v),
    ...                      dim=2, n_iter=30, seed=0)
    >>> np.linalg.norm(x) < 0.5
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def cem_optimize(
    objective: Callable[[NDArray[np.float64]], float],
    dim: int,
    *, n_samples: int = 100, elite_frac: float = 0.1,
    n_iter: int = 50, init_sigma: float = 1.0,
    seed: int | None = 0,
) -> tuple[NDArray[np.float64], float]:
    """Maximize objective (주의: CEM 은 maximize)."""
    rng = np.random.default_rng(seed)
    mu = np.zeros(dim)
    sigma = np.full(dim, init_sigma)
    best_x = mu.copy()
    best_f = float(objective(mu))
    n_elite = max(1, int(n_samples * elite_frac))

    iteration = 0
    while iteration < n_iter:
        samples = rng.normal(mu, sigma, size=(n_samples, dim))
        fvals = np.fromiter(map(lambda sample: float(objective(sample)), samples), dtype=np.float64, count=n_samples)
        order = np.argsort(-fvals)  # descending
        elite = samples[order[:n_elite]]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 1e-6)
        if fvals[order[0]] > best_f:
            best_f = float(fvals[order[0]])
            best_x = samples[order[0]].copy()
        iteration += 1
    return best_x, best_f


__all__ = ["cem_optimize"]
