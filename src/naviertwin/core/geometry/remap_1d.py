"""Conservative remapping — 1D piecewise-constant cell averages → new grid.

각 새 cell 의 평균 = 겹치는 영역 가중 합.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.geometry.remap_1d import conservative_remap_1d
    >>> x_old = np.linspace(0, 1, 5)
    >>> u_old = np.array([1.0, 2.0, 3.0, 4.0])
    >>> x_new = np.linspace(0, 1, 3)
    >>> u_new = conservative_remap_1d(x_old, u_old, x_new)
    >>> u_new.shape
    (2,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def conservative_remap_1d(
    x_old_edges: NDArray[np.float64],
    u_old: NDArray[np.float64],
    x_new_edges: NDArray[np.float64],
) -> NDArray[np.float64]:
    """piecewise-constant: u_new[i] = ∫_{x_new[i]}^{x_new[i+1]} u_old / Δx_new."""
    xo = np.asarray(x_old_edges, dtype=np.float64)
    xn = np.asarray(x_new_edges, dtype=np.float64)
    u_old = np.asarray(u_old, dtype=np.float64)
    u_new = np.zeros(len(xn) - 1)
    for i in range(len(xn) - 1):
        a, b = xn[i], xn[i + 1]
        total = 0.0
        for j in range(len(xo) - 1):
            ov_a = max(a, xo[j])
            ov_b = min(b, xo[j + 1])
            if ov_b > ov_a:
                total += (ov_b - ov_a) * u_old[j]
        u_new[i] = total / (b - a)
    return u_new


__all__ = ["conservative_remap_1d"]
