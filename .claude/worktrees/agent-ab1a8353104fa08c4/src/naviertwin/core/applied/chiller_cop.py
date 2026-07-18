"""Chiller COP map — bilinear interpolation from a (T_cw, T_chw) table.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.applied.chiller_cop import interpolate_cop
    >>> T_cw = np.array([20, 30])
    >>> T_chw = np.array([5, 10])
    >>> table = np.array([[5.0, 4.5], [4.0, 3.5]])
    >>> abs(interpolate_cop(T_cw_q=25, T_chw_q=7.5, T_cw=T_cw, T_chw=T_chw, COP=table) - 4.25) < 1e-3
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def interpolate_cop(
    *, T_cw_q: float, T_chw_q: float,
    T_cw: NDArray[np.float64], T_chw: NDArray[np.float64],
    COP: NDArray[np.float64],
) -> float:
    """Bilinear in (T_cw, T_chw)."""
    # find indices
    i = np.clip(np.searchsorted(T_cw, T_cw_q) - 1, 0, len(T_cw) - 2)
    j = np.clip(np.searchsorted(T_chw, T_chw_q) - 1, 0, len(T_chw) - 2)
    t = (T_cw_q - T_cw[i]) / (T_cw[i + 1] - T_cw[i])
    s = (T_chw_q - T_chw[j]) / (T_chw[j + 1] - T_chw[j])
    c = (
        (1 - t) * (1 - s) * COP[i, j]
        + t * (1 - s) * COP[i + 1, j]
        + (1 - t) * s * COP[i, j + 1]
        + t * s * COP[i + 1, j + 1]
    )
    return float(c)


__all__ = ["interpolate_cop"]
