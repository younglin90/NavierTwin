"""BEM (Blade Element Momentum) — wind turbine power coefficient.

CP_max ≈ Betz 0.593. Simplified solver per blade section.

Examples:
    >>> from naviertwin.core.applied.bem_turbine import betz_limit, cp_estimate
    >>> betz_limit()
    0.5925925925925926
"""

from __future__ import annotations

import numpy as np


def betz_limit() -> float:
    return 16.0 / 27.0


def cp_estimate(*, tip_speed_ratio: float, n_blades: int = 3,
                  cl: float = 1.0, cd: float = 0.05) -> float:
    """Crude CP via tip-speed-ratio + L/D heuristic."""
    eta = cl / (cl + cd) if (cl + cd) > 0 else 0
    cp = betz_limit() * eta * (1 - np.exp(-0.4 * tip_speed_ratio))
    cp = max(0.0, min(cp, betz_limit()))
    return float(cp)


__all__ = ["betz_limit", "cp_estimate"]
