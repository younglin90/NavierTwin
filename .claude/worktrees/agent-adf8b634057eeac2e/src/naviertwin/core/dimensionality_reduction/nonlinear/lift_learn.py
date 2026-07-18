"""Lift & Learn — Qian et al. 2020. Lift state into augmented vars → fit linear.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.nonlinear.lift_learn import lift_polynomial
    >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> X_lift = lift_polynomial(X)
    >>> X_lift.shape
    (2, 5)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def lift_polynomial(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """augment X with X² and X·X (cross): [X | X² | X·shift]."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    sq = X ** 2
    i_idx, j_idx = np.triu_indices(X.shape[1], k=1)
    if i_idx.size == 0:
        return np.hstack([X, sq])
    cross = X[:, i_idx] * X[:, j_idx]
    return np.hstack([X, sq, cross])


def fit_lifted_linear(
    X_lifted: NDArray, Xdot: NDArray,
) -> NDArray:
    A, *_ = np.linalg.lstsq(X_lifted, Xdot, rcond=None)
    return A.T


__all__ = ["fit_lifted_linear", "lift_polynomial"]
