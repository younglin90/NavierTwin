"""Bootstrap confidence intervals — non-parametric resampling.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.uncertainty.bootstrap import bootstrap_ci
    >>> rng = np.random.default_rng(0)
    >>> data = rng.normal(0, 1, 100)
    >>> lo, hi = bootstrap_ci(data, np.mean, n_boot=500, rng=rng)
    >>> lo < 0 < hi
    np.True_
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def bootstrap_ci(
    data: NDArray[np.float64],
    statistic: Callable[[NDArray[np.float64]], float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """percentile bootstrap CI."""
    rng = rng if rng is not None else np.random.default_rng(0)
    data = np.asarray(data)
    n = len(data)
    boot = np.empty(n_boot)
    k = 0
    while k < n_boot:
        sample = data[rng.integers(0, n, n)]
        boot[k] = float(statistic(sample))
        k += 1
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


__all__ = ["bootstrap_ci"]
