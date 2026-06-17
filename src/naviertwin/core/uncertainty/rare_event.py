"""Rare-event probability — multilevel splitting (subset simulation lite).

P(g(X) > b) when b is rare.  Subset Simulation (Au & Beck 2001).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.uncertainty.rare_event import subset_simulation
    >>> rng = np.random.default_rng(0)
    >>> p = subset_simulation(
    ...     g=lambda x: x.sum(axis=-1),
    ...     d=2, b=4.0, p0=0.1, n=200, rng=rng,
    ... )
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def subset_simulation(
    g: Callable[[np.ndarray], np.ndarray],
    *,
    d: int = 1,
    b: float = 3.0,
    p0: float = 0.1,
    n: int = 500,
    max_levels: int = 10,
    rng: np.random.Generator | None = None,
) -> float:
    """X ~ N(0, I_d), 추정 P(g(X) > b)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    X = rng.standard_normal((n, d))
    levels = []
    level = 0
    while level < max_levels:
        gv = g(X)
        if np.mean(gv > b) > p0:
            return float(np.mean(gv > b)) * np.prod([p0] * len(levels))
        thresh = float(np.quantile(gv, 1 - p0))
        levels.append(thresh)
        if thresh >= b:
            return p0 ** len(levels) * float(np.mean(gv > b))
        # seed: top p0 fraction
        seeds = X[gv >= thresh]
        # propagate via random walk MH (Gaussian proposal)
        new_X = []
        idx = 0
        while len(new_X) < n:
            x = seeds[idx % len(seeds)]
            prop = 0.5 * x + np.sqrt(1 - 0.25) * rng.standard_normal(d)
            if g(prop[None, :])[0] > thresh:
                new_X.append(prop)
            else:
                new_X.append(x)
            idx += 1
        X = np.asarray(new_X)
        level += 1
    return p0 ** max_levels


__all__ = ["subset_simulation"]
