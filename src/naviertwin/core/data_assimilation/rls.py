"""Recursive Least Squares (forgetting factor).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.data_assimilation.rls import RLS
    >>> rls = RLS(n_features=2, lam=0.99)
    >>> for x, y in [(np.array([1, 1]), 2.0), (np.array([1, 0]), 1.0)]:
    ...     rls.update(x, y)
    >>> rls.theta.shape
    (2,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class RLS:
    def __init__(self, n_features: int, *, lam: float = 0.99,
                 delta: float = 1.0) -> None:
        self.theta = np.zeros(n_features)
        self.P = delta * np.eye(n_features)
        self.lam = lam

    def update(self, x: NDArray[np.float64], y: float) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        Px = self.P @ x
        denom = self.lam + float(x @ Px)
        K = Px / denom
        e = float(y) - float(self.theta @ x)
        self.theta = self.theta + K * e
        self.P = (self.P - np.outer(K, Px)) / self.lam


__all__ = ["RLS"]
