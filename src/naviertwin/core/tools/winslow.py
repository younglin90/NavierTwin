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

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required by Winslow smoothing")


def winslow_smooth(
    X: NDArray[np.float64], Y: NDArray[np.float64], *, n_iter: int = 30,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """fixed boundary, Gauss-Seidel ∇² = 0."""
    Xs, Ys = _kernels.winslow_smooth(
        np.asarray(X, dtype=np.float64),
        np.asarray(Y, dtype=np.float64),
        int(n_iter),
    )
    return np.asarray(Xs, dtype=np.float64), np.asarray(Ys, dtype=np.float64)


__all__ = ["winslow_smooth"]
