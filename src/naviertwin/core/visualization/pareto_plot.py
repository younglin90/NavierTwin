"""Pareto front extraction (2D, minimization).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.visualization.pareto_plot import pareto_front
    >>> obj = np.array([[1, 5], [2, 3], [3, 4], [4, 1]])
    >>> mask = pareto_front(obj)
    >>> mask.tolist()
    [True, True, False, True]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels


def pareto_front(objectives: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Returns boolean mask of non-dominated points (minimization)."""
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by pareto_front")
    return _kernels.pareto_front(np.asarray(objectives, dtype=np.float64))


__all__ = ["pareto_front"]
