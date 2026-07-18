"""DEM Hertzian contact — F = (4/3) E* √R* δ^(3/2).

Examples:
    >>> from naviertwin.core.meshless.dem_hertz import hertz_force
    >>> hertz_force(delta=0.001, E_star=1e9, R_star=0.01) > 0
    True
"""

from __future__ import annotations

import numpy as np


def hertz_force(*, delta: float, E_star: float, R_star: float) -> float:
    if delta <= 0:
        return 0.0
    return float((4.0 / 3.0) * E_star * np.sqrt(R_star) * delta ** 1.5)


def E_star(*, E1: float, nu1: float, E2: float, nu2: float) -> float:
    return float(1.0 / ((1 - nu1 ** 2) / E1 + (1 - nu2 ** 2) / E2))


def R_star(*, R1: float, R2: float) -> float:
    return float(R1 * R2 / (R1 + R2))


__all__ = ["E_star", "R_star", "hertz_force"]
