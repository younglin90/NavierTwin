"""Anisotropic adaptation driver — Hessian-based metric scaling.

target N_target cells → scale metric by α s.t. Σ √det(M) = N_target.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.amr.aniso_driver import normalize_metric
    >>> M = np.array([np.eye(2), 2*np.eye(2)])
    >>> M2 = normalize_metric(M, n_target=10)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def normalize_metric(
    M: NDArray[np.float64], n_target: int = 100,
) -> NDArray[np.float64]:
    """α^d Σ √det(M) = N_target → scale (d=2)."""
    M = np.asarray(M, dtype=np.float64)
    d = M.shape[-1]
    dets = np.asarray(_kernels.determinant_batch(M), dtype=np.float64)
    sqrt_dets = np.sqrt(np.maximum(dets, 0))
    s = sqrt_dets.sum()
    if s <= 0:
        return M
    alpha = (n_target / s) ** (2.0 / d)
    return M * alpha


__all__ = ["normalize_metric"]
