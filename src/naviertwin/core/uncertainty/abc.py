"""Approximate Bayesian Computation — rejection-ABC.

샘플 후보를 prior 에서 뽑아 d(simulator(θ), y_obs) < ε 만 채택.

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> def sim(theta): return rng.normal(theta, 1)
    >>> from naviertwin.core.uncertainty.abc import abc_rejection
    >>> samples = abc_rejection(sim, y_obs=2.0, prior=lambda r: r.uniform(-5, 5),
    ...                          eps=0.5, n_target=50, max_iter=10000, rng=rng)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray


def abc_rejection(
    simulator: Callable[[Any], float],
    y_obs: float,
    prior: Callable[[np.random.Generator], Any],
    *,
    eps: float = 0.1,
    n_target: int = 100,
    max_iter: int = 100_000,
    rng: np.random.Generator | None = None,
) -> NDArray:
    rng = rng if rng is not None else np.random.default_rng(0)
    accepted: list[Any] = []
    it = 0
    while it < max_iter:
        if len(accepted) >= n_target:
            break
        theta = prior(rng)
        y = simulator(theta)
        if abs(y - y_obs) < eps:
            accepted.append(theta)
        it += 1
    return np.asarray(accepted)


__all__ = ["abc_rejection"]
