"""CFD 색맵 유틸 — scalar → RGB 매핑.

matplotlib 없이 동작하는 built-in 맵(viridis, plasma, turbo, coolwarm, jet).

Examples:
    >>> import numpy as np
    >>> from naviertwin.gui.colormaps import apply_colormap
    >>> rgb = apply_colormap(np.array([0.0, 0.5, 1.0]), "viridis")
    >>> rgb.shape
    (3, 3)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# 각 맵: 제어점 (value in [0,1], R, G, B) 선형 보간
_MAPS: dict[str, list[tuple[float, float, float, float]]] = {
    "viridis": [
        (0.00, 0.267, 0.004, 0.329),
        (0.25, 0.229, 0.322, 0.545),
        (0.50, 0.127, 0.566, 0.550),
        (0.75, 0.369, 0.788, 0.382),
        (1.00, 0.992, 0.906, 0.143),
    ],
    "plasma": [
        (0.00, 0.050, 0.029, 0.527),
        (0.25, 0.451, 0.000, 0.658),
        (0.50, 0.799, 0.278, 0.469),
        (0.75, 0.987, 0.602, 0.227),
        (1.00, 0.940, 0.975, 0.131),
    ],
    "turbo": [
        (0.00, 0.190, 0.072, 0.232),
        (0.25, 0.174, 0.569, 0.894),
        (0.50, 0.441, 0.938, 0.390),
        (0.75, 0.945, 0.571, 0.201),
        (1.00, 0.479, 0.015, 0.011),
    ],
    "coolwarm": [
        (0.00, 0.230, 0.299, 0.754),
        (0.50, 0.865, 0.865, 0.865),
        (1.00, 0.706, 0.016, 0.150),
    ],
    "jet": [
        (0.00, 0.0, 0.0, 0.5),
        (0.125, 0.0, 0.0, 1.0),
        (0.375, 0.0, 1.0, 1.0),
        (0.625, 1.0, 1.0, 0.0),
        (0.875, 1.0, 0.0, 0.0),
        (1.00, 0.5, 0.0, 0.0),
    ],
}


def available_colormaps() -> list[str]:
    return sorted(_MAPS.keys())


def apply_colormap(
    values: NDArray[np.float64], name: str = "viridis",
    *, vmin: float | None = None, vmax: float | None = None,
) -> NDArray[np.float64]:
    """(...,) → (..., 3) RGB [0,1]."""
    if name not in _MAPS:
        raise ValueError(f"unknown colormap: {name}. available: {available_colormaps()}")
    v = np.asarray(values, dtype=np.float64)
    vmin = float(v.min()) if vmin is None else float(vmin)
    vmax = float(v.max()) if vmax is None else float(vmax)
    if vmax == vmin:
        vmax = vmin + 1.0
    t = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)
    stops = _MAPS[name]
    ts, r, g, b = np.asarray(stops, dtype=np.float64).T
    R = np.interp(t, ts, r)
    G = np.interp(t, ts, g)
    B = np.interp(t, ts, b)
    return np.stack([R, G, B], axis=-1)


__all__ = ["available_colormaps", "apply_colormap"]
