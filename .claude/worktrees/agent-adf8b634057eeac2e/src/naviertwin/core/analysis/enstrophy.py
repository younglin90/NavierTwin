"""Enstrophy ζ = (1/2) |ω|².

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.enstrophy import enstrophy_density
    >>> w = np.array([[2.0, 0, 0]])
    >>> enstrophy_density(w)
    array([2.])
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def enstrophy_density(omega: NDArray[np.float64]) -> NDArray[np.float64]:
    """ζ = (1/2) |ω|², 마지막 축 = 3."""
    w = np.asarray(omega, dtype=np.float64)
    return 0.5 * np.sum(w * w, axis=-1)


def integrated_enstrophy(omega: NDArray[np.float64], dV: float = 1.0) -> float:
    return float(enstrophy_density(omega).sum() * dV)


__all__ = ["enstrophy_density", "integrated_enstrophy"]
