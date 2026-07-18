"""Jackknife variance estimator — leave-one-out.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.uncertainty.jackknife import jackknife_var
    >>> data = np.array([1.0, 2.0, 3.0, 4.0])
    >>> v = jackknife_var(data, np.mean)
    >>> v > 0
    np.True_
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required by jackknife estimators")


def jackknife_var(
    data: NDArray[np.float64], statistic: Callable[[NDArray[np.float64]], float],
) -> float:
    """jackknife variance: ((n-1)/n) Σ (θ_i - θ_dot)²."""
    data = np.asarray(data, dtype=np.float64)
    if statistic is np.mean:
        return float(_kernels.jackknife_mean_var(data.ravel()))
    n = len(data)
    theta_i = np.fromiter(
        map(lambda i: float(statistic(np.delete(data, i))), range(n)),
        dtype=np.float64,
        count=n,
    )
    theta_dot = theta_i.mean()
    return (n - 1) / n * float(np.sum((theta_i - theta_dot) ** 2))


def jackknife_se(
    data: NDArray[np.float64], statistic: Callable[[NDArray[np.float64]], float],
) -> float:
    return float(np.sqrt(jackknife_var(data, statistic)))


__all__ = ["jackknife_se", "jackknife_var"]
