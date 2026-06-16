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

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def dtw_distance(
    a: NDArray[np.float64], b: NDArray[np.float64],
    *, window: int | None = None,
) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(_kernels.dtw_distance(a, b, window))


def dtw_matrix(a, b) -> NDArray[np.float64]:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.asarray(_kernels.dtw_matrix(a, b), dtype=np.float64)


__all__ = ["dtw_distance", "dtw_matrix"]
