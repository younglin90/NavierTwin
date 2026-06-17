"""Ghost-Fluid Method (GFM) — Fedkiw 1999, jump-condition aware extrapolation.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.geometry.ghost_fluid import gfm_extend
    >>> u = np.array([1.0, 1.0, 0.0, 0.0])
    >>> phi = np.array([-1, -1, 1, 1], dtype=float)
    >>> u_ext = gfm_extend(u, phi)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def gfm_extend(
    u: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """간단 1D GFM: φ<0 (fluid 1) 영역 값을 φ>0 (fluid 2) ghost 로 복사."""
    u = np.asarray(u, dtype=np.float64).copy()
    phi = np.asarray(phi, dtype=np.float64)
    out = u.copy()
    fluid1_idx = np.where(phi < 0)[0]
    if fluid1_idx.size == 0:
        return out
    ghost_idx = np.where(phi > 0)[0]
    if ghost_idx.size == 0:
        return out
    right_pos = np.searchsorted(fluid1_idx, ghost_idx, side="left")
    left_pos = np.clip(right_pos - 1, 0, fluid1_idx.size - 1)
    right_pos = np.clip(right_pos, 0, fluid1_idx.size - 1)
    left_idx = fluid1_idx[left_pos]
    right_idx = fluid1_idx[right_pos]
    choose_left = np.abs(ghost_idx - left_idx) <= np.abs(right_idx - ghost_idx)
    nearest_idx = np.where(choose_left, left_idx, right_idx)
    out[ghost_idx] = u[nearest_idx]
    return out


__all__ = ["gfm_extend"]
