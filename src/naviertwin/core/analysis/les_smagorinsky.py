"""LES Smagorinsky — sub-grid eddy viscosity ν_sgs = (Cs Δ)² |S|.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.les_smagorinsky import smagorinsky_nu_sgs
    >>> S = np.ones((3, 3))
    >>> nu = smagorinsky_nu_sgs(S, dx=0.1)
    >>> nu.shape
    (3, 3)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

CS_DEFAULT = 0.17


def smagorinsky_nu_sgs(
    strain_rate_mag: NDArray[np.float64], dx: float, *, Cs: float = CS_DEFAULT,
) -> NDArray[np.float64]:
    """ν_sgs = (Cs Δ)² |S|, |S| = √(2 S_ij S_ij)."""
    S = np.asarray(strain_rate_mag, dtype=np.float64)
    return (Cs * dx) ** 2 * S


def filter_box_2d(
    field: NDArray[np.float64], width: int = 3,
) -> NDArray[np.float64]:
    """Box filter (uniform) — 2D."""
    f = np.asarray(field, dtype=np.float64)
    pad = width // 2
    fp = np.pad(f, pad, mode="edge")
    out = np.zeros_like(f)
    for i in range(width):
        for j in range(width):
            out += fp[i:i + f.shape[0], j:j + f.shape[1]]
    return out / (width * width)


__all__ = ["smagorinsky_nu_sgs", "filter_box_2d", "CS_DEFAULT"]
