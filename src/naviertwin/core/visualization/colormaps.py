"""Color map library — viridis-lite, jet, gray, coolwarm.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.visualization.colormaps import apply_cmap
    >>> rgb = apply_cmap(np.linspace(0, 1, 5), name='viridis')
    >>> rgb.shape
    (5, 3)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# coarse-tabulated viridis (5 stops) — interp at runtime
_VIRIDIS = np.array([
    [0.267, 0.005, 0.329],
    [0.230, 0.322, 0.546],
    [0.128, 0.567, 0.551],
    [0.369, 0.788, 0.383],
    [0.993, 0.906, 0.144],
])

_JET = np.array([
    [0.0, 0.0, 0.5],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
])

_COOLWARM = np.array([
    [0.230, 0.299, 0.754],
    [0.706, 0.706, 0.706],
    [0.706, 0.016, 0.150],
])


def _table(name: str) -> NDArray[np.float64]:
    return {
        "viridis": _VIRIDIS,
        "jet": _JET,
        "coolwarm": _COOLWARM,
    }[name]


def apply_cmap(values: NDArray[np.float64], *, name: str = "viridis") -> NDArray[np.float64]:
    """values in [0, 1] → (N, 3) RGB."""
    if name == "gray":
        v = np.clip(np.asarray(values), 0, 1)
        return np.stack([v, v, v], axis=-1)
    table = _table(name)
    v = np.clip(np.asarray(values, dtype=np.float64), 0, 1)
    n_stops = table.shape[0] - 1
    pos = v * n_stops
    i = np.clip(pos.astype(int), 0, n_stops - 1)
    f = pos - i
    rgb = table[i] * (1 - f[..., None]) + table[i + 1] * f[..., None]
    return rgb


__all__ = ["apply_cmap"]
