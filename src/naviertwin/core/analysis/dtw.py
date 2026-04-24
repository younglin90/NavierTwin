"""Dynamic Time Warping — 시계열 유사도 (align cost).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.dtw import dtw_distance
    >>> a = np.array([1., 2., 3., 2., 1.])
    >>> b = np.array([1., 1., 2., 3., 2., 1.])
    >>> d = dtw_distance(a, b)
    >>> d < 1.0
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def dtw_distance(
    a: NDArray[np.float64], b: NDArray[np.float64],
    *, window: int | None = None,
) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n, m = a.size, b.size
    w = window if window is not None else max(n, m)
    INF = np.inf
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        jlo = max(1, i - w)
        jhi = min(m, i + w)
        for j in range(jlo, jhi + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def dtw_matrix(a, b) -> NDArray[np.float64]:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n, m = a.size, b.size
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = abs(a[i - 1] - b[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D


__all__ = ["dtw_distance", "dtw_matrix"]
