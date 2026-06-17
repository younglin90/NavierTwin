"""Manifold tangent space — local PCA at point.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.nonlinear.manifold_tangent import (
    ...     tangent_space,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 5))
    >>> T = tangent_space(X, p_idx=0, k=20, dim=2)
    >>> T.shape
    (5, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray


def tangent_space(
    X: NDArray[np.float64], *, p_idx: int, k: int = 20, dim: int = 2,
) -> NDArray[np.float64]:
    """k-NN around X[p_idx] → SVD → top `dim` singular vectors."""
    X = np.asarray(X, dtype=np.float64)
    p = X[p_idx]
    d = np.linalg.norm(X - p, axis=1)
    nn = np.argsort(d)[:k]
    Y = X[nn] - p
    U, _, _ = _svd(Y.T, full_matrices=False)
    return U[:, :dim]


__all__ = ["tangent_space"]
