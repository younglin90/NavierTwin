"""Time-series plot helper — return plot-ready arrays (no matplotlib import).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.visualization.timeseries_plot import series_with_band
    >>> t = np.linspace(0, 1, 5); y = np.zeros(5); s = np.ones(5)
    >>> data = series_with_band(t, y, s)
    >>> 'upper' in data and 'lower' in data
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def series_with_band(
    t: NDArray[np.float64], y: NDArray[np.float64], std: NDArray[np.float64], *,
    z: float = 1.96,
) -> dict:
    t = np.asarray(t)
    y = np.asarray(y)
    s = np.asarray(std)
    return {
        "t": t, "mean": y,
        "upper": y + z * s, "lower": y - z * s,
    }


def downsample(
    t: NDArray[np.float64], y: NDArray[np.float64], *, max_points: int = 1000,
) -> tuple[NDArray, NDArray]:
    n = len(t)
    if n <= max_points:
        return t, y
    step = n // max_points
    return t[::step], y[::step]


__all__ = ["downsample", "series_with_band"]
