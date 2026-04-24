"""Space-time POD — 공간-시간 결합 stack 위에서 POD.

X (n, m) 시계열을 (window, n)·sliding 으로 reshape 후 SVD.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.linear.space_time_pod import (
    ...     SpaceTimePOD,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((20, 50))
    >>> stp = SpaceTimePOD(window=5, rank=3).fit(X)
    >>> stp.modes.shape
    (100, 3)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SpaceTimePOD:
    def __init__(self, window: int = 5, rank: int = 5) -> None:
        self.window = int(window)
        self.rank = int(rank)
        self.modes: NDArray | None = None
        self.singular_values: NDArray | None = None

    def fit(self, X: NDArray[np.float64]) -> "SpaceTimePOD":
        X = np.asarray(X, dtype=np.float64)
        n, m = X.shape
        w = self.window
        if m < w:
            raise ValueError("snapshots must be >= window")
        # build space-time matrix: each column = stacked w consecutive snapshots
        cols = []
        for k in range(m - w + 1):
            cols.append(X[:, k:k + w].reshape(-1, order="F"))
        Y = np.column_stack(cols)  # (n*w, m-w+1)
        U, s, _ = np.linalg.svd(Y, full_matrices=False)
        r = min(self.rank, U.shape[1])
        self.modes = U[:, :r]
        self.singular_values = s[:r]
        return self

    def project(self, segment: NDArray[np.float64]) -> NDArray[np.float64]:
        """segment: (n, window) → coefficients (rank,)."""
        v = np.asarray(segment, dtype=np.float64).reshape(-1, order="F")
        return self.modes.T @ v


__all__ = ["SpaceTimePOD"]
