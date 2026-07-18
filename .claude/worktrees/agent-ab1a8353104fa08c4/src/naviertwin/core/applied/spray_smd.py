"""Spray Sauter mean diameter (SMD = D32) — Σ d³ / Σ d².

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.applied.spray_smd import sauter_mean
    >>> sauter_mean(np.array([1.0, 2.0, 3.0]))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sauter_mean(diameters: NDArray[np.float64]) -> float:
    d = np.asarray(diameters, dtype=np.float64)
    num = float(np.sum(d ** 3))
    den = float(np.sum(d ** 2))
    return num / max(den, 1e-30)


__all__ = ["sauter_mean"]
