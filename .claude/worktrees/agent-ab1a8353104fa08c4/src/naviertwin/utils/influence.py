"""Influence function — leave-one-out effect on parameter (linear regression).

I_i = -H^{-1} ∇_θ L(z_i, θ).

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.influence import linreg_influence
    >>> X = np.eye(3); y = np.array([1., 2., 3.])
    >>> infl = linreg_influence(X, y)
    >>> infl.shape
    (3, 3)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def linreg_influence(
    X: NDArray[np.float64], y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Returns matrix (n, d) of influence of each sample on parameter."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    H = X.T @ X + 1e-10 * np.eye(X.shape[1])
    Hinv = np.linalg.inv(H)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    projected = X @ Hinv.T
    if res.ndim == 1:
        return projected * res[:, None]
    return projected[:, :, None] * res[:, None, :]


__all__ = ["linreg_influence"]
