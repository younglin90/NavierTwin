"""Multi-Level Monte Carlo (Giles 2008).

E[f_L] = E[f_0] + Σ_{l=1}^L E[f_l - f_{l-1}].  각 레벨별 비용 차이 활용.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.uncertainty.mlmc import mlmc_estimate
    >>> rng = np.random.default_rng(0)
    >>> def sample(level, rng):
    ...     return rng.standard_normal()  # toy
    >>> est, var = mlmc_estimate(sample, levels=[100, 50, 20, 10], rng=rng)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def mlmc_estimate(
    level_sampler: Callable[[int, np.random.Generator], float],
    *,
    levels: list[int],
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """level_sampler(level, rng) → 보통 f_l - f_{l-1} sample (level=0 은 f_0).

    levels[l] = 해당 레벨 샘플 수.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    est = 0.0
    var = 0.0
    for lvl, n_l in enumerate(levels):
        samples = np.array([level_sampler(lvl, rng) for _ in range(n_l)])
        est += samples.mean()
        var += samples.var(ddof=1) / max(n_l, 1)
    return float(est), float(var)


__all__ = ["mlmc_estimate"]
