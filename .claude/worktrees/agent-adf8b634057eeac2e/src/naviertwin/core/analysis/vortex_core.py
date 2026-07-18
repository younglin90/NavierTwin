"""Vortex core line — swirl-strength λ_ci (Zhou et al. 1999).

∇u 의 복소 고유치의 허수부 = swirl strength. λ_ci > 0 → vortex.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.vortex_core import swirl_strength_field
    >>> grad = np.zeros((1, 3, 3))
    >>> grad[0] = [[0, -1, 0], [1, 0, 0], [0, 0, 0]]
    >>> swirl_strength_field(grad)[0] > 0
    np.True_
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def swirl_strength_field(
    grad_u: NDArray[np.float64],
) -> NDArray[np.float64]:
    """grad_u shape (..., 3, 3) → swirl strength (...,)."""
    g = np.asarray(grad_u, dtype=np.float64)
    flat = g.reshape(-1, 3, 3)
    ev = np.linalg.eigvals(flat)
    out = np.max(np.abs(ev.imag), axis=1)
    return out.reshape(g.shape[:-2])


__all__ = ["swirl_strength_field"]
