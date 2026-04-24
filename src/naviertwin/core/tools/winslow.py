"""Winslow elliptic mesh smoothing (2D structured grid).

(x, y) 함수가 isothermal: ∇²ξ = 0, ∇²η = 0  →  Gauss-Seidel 갱신.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.tools.winslow import winslow_smooth
    >>> rng = np.random.default_rng(0)
    >>> X = np.linspace(0, 1, 5)[:, None] * np.ones(5)
    >>> Y = np.ones(5)[:, None] * np.linspace(0, 1, 5)
    >>> X[2, 2] += 0.1; Y[2, 2] += 0.1
    >>> Xs, Ys = winslow_smooth(X, Y, n_iter=10)
    >>> Xs.shape
    (5, 5)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def winslow_smooth(
    X: NDArray[np.float64], Y: NDArray[np.float64], *, n_iter: int = 30,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """fixed boundary, Gauss-Seidel ∇² = 0."""
    X = np.asarray(X, dtype=np.float64).copy()
    Y = np.asarray(Y, dtype=np.float64).copy()
    for _ in range(n_iter):
        X[1:-1, 1:-1] = 0.25 * (
            X[2:, 1:-1] + X[:-2, 1:-1] + X[1:-1, 2:] + X[1:-1, :-2]
        )
        Y[1:-1, 1:-1] = 0.25 * (
            Y[2:, 1:-1] + Y[:-2, 1:-1] + Y[1:-1, 2:] + Y[1:-1, :-2]
        )
    return X, Y


__all__ = ["winslow_smooth"]
