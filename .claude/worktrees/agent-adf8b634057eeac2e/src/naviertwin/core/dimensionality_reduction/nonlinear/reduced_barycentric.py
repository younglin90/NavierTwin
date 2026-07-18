"""Reduced barycentric coordinates — express x ≈ Σ w_k v_k, w on simplex.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.nonlinear.reduced_barycentric import (
    ...     barycentric_weights,
    ... )
    >>> V = np.array([[0., 0], [1., 0], [0., 1.]])  # 3 anchors in 2D
    >>> w = barycentric_weights(np.array([0.3, 0.3]), V)
    >>> abs(w.sum() - 1) < 1e-6
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def barycentric_weights(
    x: NDArray[np.float64], anchors: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve V w = x, Σ w = 1 with non-negativity (project onto simplex)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    V = np.asarray(anchors, dtype=np.float64)
    # Augment: [V; 1ᵀ] w = [x; 1] (least squares)
    A = np.vstack([V.T, np.ones(len(V))])
    b = np.concatenate([x, [1.0]])
    w, *_ = np.linalg.lstsq(A, b, rcond=None)
    # project to simplex: clip non-neg, renorm
    w = np.maximum(w, 0)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


__all__ = ["barycentric_weights"]
