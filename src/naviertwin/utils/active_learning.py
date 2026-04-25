"""Active learning query strategies — uncertainty / margin / random.

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.active_learning import uncertainty_query
    >>> probs = np.array([[0.9, 0.1], [0.55, 0.45]])
    >>> uncertainty_query(probs, k=1).tolist()
    [1]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def uncertainty_query(probs: NDArray[np.float64], k: int = 10) -> NDArray[np.int_]:
    """Smallest top-1 confidence."""
    p = np.asarray(probs)
    top = p.max(axis=1)
    return np.argsort(top)[:k]


def margin_query(probs: NDArray[np.float64], k: int = 10) -> NDArray[np.int_]:
    """Smallest margin between top-1 and top-2."""
    p = np.asarray(probs)
    sort = np.sort(p, axis=1)
    margin = sort[:, -1] - sort[:, -2]
    return np.argsort(margin)[:k]


def random_query(n: int, k: int = 10, seed: int = 0) -> NDArray[np.int_]:
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=min(k, n), replace=False)


__all__ = ["margin_query", "random_query", "uncertainty_query"]
