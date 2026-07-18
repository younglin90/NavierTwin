"""Dual-weighted residual (DWR) — Becker & Rannacher 2001.

η = (R(u_h), z_h - z_h^lower) — residual weighted by dual interpolation gap.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.amr.dwr import dwr_indicator
    >>> R = np.array([0.1, 0.5])
    >>> z_high = np.array([1.0, 1.0])
    >>> z_low = np.array([0.5, 0.9])
    >>> dwr_indicator(R, z_high, z_low)
    array([0.05, 0.05])
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def dwr_indicator(
    residual: NDArray[np.float64],
    dual_high: NDArray[np.float64],
    dual_low: NDArray[np.float64],
) -> NDArray[np.float64]:
    """η_K = R_K * (z_high - z_low)_K."""
    return np.asarray(residual) * (np.asarray(dual_high) - np.asarray(dual_low))


__all__ = ["dwr_indicator"]
