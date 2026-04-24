"""Helicity h = u · ω.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.helicity import helicity_density
    >>> u = np.array([[1.0, 0, 0]])
    >>> w = np.array([[1.0, 0, 0]])
    >>> helicity_density(u, w)
    array([1.])
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def helicity_density(
    u: NDArray[np.float64], omega: NDArray[np.float64],
) -> NDArray[np.float64]:
    """h = u · ω, 마지막 축 = 3 (vector dim)."""
    u = np.asarray(u, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    return np.sum(u * omega, axis=-1)


def integrated_helicity(
    u: NDArray[np.float64], omega: NDArray[np.float64], dV: float = 1.0,
) -> float:
    """∫ u · ω dV."""
    return float(helicity_density(u, omega).sum() * dV)


__all__ = ["helicity_density", "integrated_helicity"]
