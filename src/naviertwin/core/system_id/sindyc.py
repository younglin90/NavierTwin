"""SINDy with control (SINDYc) — Brunton 2016.

ẋ = Θ([x, u]) Ξ.  Library on stacked features.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.system_id.sindyc import sindyc_fit
    >>> X = np.linspace(0, 1, 50)[:, None]
    >>> Xdot = -X
    >>> U = np.zeros_like(X)
    >>> Xi = sindyc_fit(X, Xdot, U)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _library(X, U):
    return np.column_stack([np.ones(len(X)), X, U, X * U, X ** 2])


def sindyc_fit(
    X: NDArray[np.float64],
    Xdot: NDArray[np.float64],
    U: NDArray[np.float64],
    *,
    threshold: float = 0.01,
    n_iter: int = 10,
) -> NDArray[np.float64]:
    """STLSQ on Θ([X, U]) → Xdot."""
    X = np.atleast_2d(np.asarray(X)).reshape(len(X), -1)
    U = np.atleast_2d(np.asarray(U)).reshape(len(U), -1)
    Theta = _library(X, U)
    Xi, *_ = np.linalg.lstsq(Theta, Xdot, rcond=None)
    it = 0
    while it < n_iter:
        small = np.abs(Xi) < threshold
        Xi[small] = 0
        j = 0
        width = Xi.shape[1] if Xi.ndim > 1 else 1
        while j < width:
            big = ~small[:, j] if Xi.ndim > 1 else ~small
            if big.any():
                xi_j, *_ = np.linalg.lstsq(
                    Theta[:, big], Xdot[:, j] if Xi.ndim > 1 else Xdot, rcond=None,
                )
                if Xi.ndim > 1:
                    Xi[big, j] = xi_j
                else:
                    Xi[big] = xi_j
            j += 1
        it += 1
    return Xi


__all__ = ["sindyc_fit"]
