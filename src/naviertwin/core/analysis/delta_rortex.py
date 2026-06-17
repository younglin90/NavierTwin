"""Δ-criterion (Chong 1990) and Rortex (Liu 2018).

Δ = (Q/3)³ + (R/2)², Δ > 0 → 복소 conjugate eigenvalues → vortex.
Rortex = 2 |λ_ci| (대략 swirl strength 의 2배).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.delta_rortex import delta_criterion
    >>> grad = np.zeros((1, 3, 3))
    >>> grad[0] = [[0, -1, 0], [1, 0, 0], [0, 0, 0]]
    >>> delta_criterion(grad)[0] > 0
    np.True_
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels


def delta_criterion(grad_u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Δ = (Q/3)³ + (R/2)²."""
    if _kernels is None:
        raise ImportError("naviertwin._native._kernels is required by delta_criterion")
    return _kernels.delta_criterion_3x3(np.asarray(grad_u, dtype=np.float64))


def rortex_field(grad_u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rortex magnitude ≈ 2 λ_ci."""
    g = np.asarray(grad_u, dtype=np.float64)
    flat = g.reshape(-1, 3, 3)
    out = np.zeros(flat.shape[0])
    i = 0
    while i < flat.shape[0]:
        ev = np.linalg.eigvals(flat[i])
        out[i] = 2.0 * float(np.max(np.abs(ev.imag)))
        i += 1
    return out.reshape(g.shape[:-2])


__all__ = ["delta_criterion", "rortex_field"]
