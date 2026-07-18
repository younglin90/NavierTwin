"""Compressor map fit — pressure ratio vs corrected mass flow (poly).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.applied.compressor_map import fit_map, predict_pr
    >>> mdot = np.array([1, 2, 3, 4.])
    >>> pr = np.array([2., 2.5, 2.7, 2.4])
    >>> coef = fit_map(mdot, pr, deg=2)
    >>> predict_pr(coef, np.array([2.5]))[0] > 2
    np.True_
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def fit_map(
    mdot: NDArray[np.float64], pressure_ratio: NDArray[np.float64],
    *, deg: int = 2,
) -> NDArray[np.float64]:
    return np.polyfit(np.asarray(mdot), np.asarray(pressure_ratio), deg)


def predict_pr(coef: NDArray, mdot: NDArray) -> NDArray:
    return np.polyval(coef, np.asarray(mdot))


def surge_margin(*, op_mdot: float, op_pr: float, surge_pr: float) -> float:
    if op_pr <= 0:
        return 0.0
    return float((surge_pr - op_pr) / op_pr)


__all__ = ["fit_map", "predict_pr", "surge_margin"]
