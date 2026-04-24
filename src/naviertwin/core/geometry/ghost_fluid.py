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
    u: NDArray[np.float64], phi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """간단 1D GFM: φ<0 (fluid 1) 영역 값을 φ>0 (fluid 2) ghost 로 복사."""
    u = np.asarray(u, dtype=np.float64).copy()
    phi = np.asarray(phi, dtype=np.float64)
    # for each cell with phi > 0, copy nearest fluid-1 value
    out = u.copy()
    fluid1_idx = np.where(phi < 0)[0]
    if fluid1_idx.size == 0:
        return out
    for i in range(len(u)):
        if phi[i] > 0:
            j = fluid1_idx[np.argmin(np.abs(fluid1_idx - i))]
            out[i] = u[j]
    return out


__all__ = ["gfm_extend"]
