"""Chebyshev collocation — 2D tensor product (Trefethen Spectral Methods style).

x_j = cos(jπ/N), j=0..N.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solvers.cheb_2d import cheb_diff_matrix
    >>> D, x = cheb_diff_matrix(8)
    >>> D.shape, x.shape
    ((9, 9), (9,))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cheb_diff_matrix(N: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """1D Chebyshev differentiation matrix (Trefethen, cheb.m)."""
    if N == 0:
        return np.zeros((1, 1)), np.array([1.0])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = (np.outer(c, 1.0 / c)) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return D, x


def laplacian_2d(N: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """∇² on (N+1)x(N+1) Chebyshev grid (Kronecker tensor product)."""
    D, x = cheb_diff_matrix(N)
    D2 = D @ D
    Im = np.eye(N + 1)
    L = np.kron(Im, D2) + np.kron(D2, Im)
    return L, x


__all__ = ["cheb_diff_matrix", "laplacian_2d"]
