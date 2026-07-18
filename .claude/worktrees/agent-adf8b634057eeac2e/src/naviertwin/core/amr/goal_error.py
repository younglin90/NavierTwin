"""Goal-oriented error estimator — η_K = |R_K · z_K| (residual · dual).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.amr.goal_error import goal_error
    >>> R = np.array([1.0, -2.0, 0.5])
    >>> z = np.array([0.5, 0.5, 1.0])
    >>> goal_error(R, z)
    array([0.5, 1. , 0.5])
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def goal_error(
    residual: NDArray[np.float64], dual: NDArray[np.float64],
) -> NDArray[np.float64]:
    """element-wise |R_K * z_K|."""
    return np.abs(np.asarray(residual) * np.asarray(dual))


def total_error(residual: NDArray, dual: NDArray) -> float:
    return float(goal_error(residual, dual).sum())


__all__ = ["goal_error", "total_error"]
