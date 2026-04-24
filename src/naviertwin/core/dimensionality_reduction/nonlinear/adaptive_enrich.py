"""Adaptive ROM basis enrichment — orthonormal augment from residual snapshots.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.nonlinear.adaptive_enrich import (
    ...     enrich_basis,
    ... )
    >>> Phi = np.eye(5)[:, :2]
    >>> r = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    >>> Phi2 = enrich_basis(Phi, r)
    >>> Phi2.shape
    (5, 3)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def enrich_basis(
    Phi: NDArray[np.float64], residual: NDArray[np.float64],
    *, eps: float = 1e-10,
) -> NDArray[np.float64]:
    """Gram-Schmidt: r ⊥ Phi → unit → augment."""
    Phi = np.asarray(Phi, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    proj = Phi.T @ r
    r_perp = r - Phi @ proj
    n = np.linalg.norm(r_perp)
    if n < eps:
        return Phi
    return np.column_stack([Phi, r_perp / n])


__all__ = ["enrich_basis"]
