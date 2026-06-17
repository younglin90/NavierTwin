"""Nested sampling (Skilling 2006) — 간단 1D version.

Live points → 가장 낮은 likelihood 제거 → prior 에서 새 점 (constrained likelihood) 추출.
근사: 간단히 prior 재추출 후 채택률 사용.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.uncertainty.nested_sampling import nested_sample
    >>> rng = np.random.default_rng(0)
    >>> samples, logZ = nested_sample(
    ...     loglike=lambda x: -0.5 * x**2,
    ...     prior_sample=lambda r: r.uniform(-5, 5),
    ...     n_live=50, n_iter=100, rng=rng,
    ... )
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray


def nested_sample(
    loglike: Callable[[Any], float],
    prior_sample: Callable[[np.random.Generator], Any],
    *,
    n_live: int = 50,
    n_iter: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray, float]:
    """반환: (dead points, logZ 추정)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    live = list(map(lambda _: prior_sample(rng), range(n_live)))
    live_ll = np.fromiter(map(loglike, live), dtype=np.float64, count=n_live)
    dead = []
    log_w = -np.log(n_live)
    log_Z = -np.inf
    k = 0
    while k < n_iter:
        idx = int(np.argmin(live_ll))
        ll_min = float(live_ll[idx])
        log_X = -(k + 1) / n_live  # log prior volume
        contribution = ll_min + log_X
        log_Z = np.logaddexp(log_Z, contribution + log_w)
        dead.append(live[idx])
        # replace with new sample exceeding ll_min
        attempt = 0
        while attempt < 1000:
            cand = prior_sample(rng)
            if loglike(cand) > ll_min:
                live[idx] = cand
                live_ll[idx] = loglike(cand)
                break
            attempt += 1
        k += 1
    return np.asarray(dead), float(log_Z)


__all__ = ["nested_sample"]
