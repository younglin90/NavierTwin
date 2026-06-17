"""Lipschitz constant estimator — sample-based + spectral-norm product.

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.lipschitz import lipschitz_sampled
    >>> def f(x): return 2.0 * x
    >>> lipschitz_sampled(f, x_samples=np.linspace(-1, 1, 50))
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def lipschitz_sampled(
    f: Callable[[NDArray], NDArray], x_samples: NDArray[np.float64],
    *, n_pairs: int = 1000, seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    xs = np.asarray(x_samples, dtype=np.float64)
    n = len(xs)
    L = 0.0
    pairs = rng.integers(0, n, (int(n_pairs), 2))
    pair_idx = 0
    while pair_idx < len(pairs):
        i, j = pairs[pair_idx]
        if i == j:
            pair_idx += 1
            continue
        a = xs[i]
        b = xs[j]
        denom = np.linalg.norm(np.atleast_1d(b) - np.atleast_1d(a)) + 1e-30
        num = np.linalg.norm(np.atleast_1d(f(b)) - np.atleast_1d(f(a)))
        L = max(L, num / denom)
        pair_idx += 1
    return float(L)


__all__ = ["lipschitz_sampled"]
