"""Chebyshev spectral 미분 + Gauss-Lobatto nodes.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.numerics.chebyshev import (
    ...     chebyshev_points, chebyshev_diff_matrix,
    ... )
    >>> x = chebyshev_points(8)
    >>> D = chebyshev_diff_matrix(8)
    >>> u = x ** 2
    >>> du = D @ u  # 예상: 2x
    >>> np.allclose(du, 2 * x, atol=1e-10)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def chebyshev_points(N: int) -> NDArray[np.float64]:
    """Chebyshev-Gauss-Lobatto 점들 x_k = cos(kπ/N), k=0..N.

    Returns:
        (N+1,) [1, ..., -1].
    """
    if N < 1:
        raise ValueError("N >= 1 필요")
    return np.cos(np.pi * np.arange(N + 1) / N)


def chebyshev_diff_matrix(N: int) -> NDArray[np.float64]:
    """스펙트럴 미분 행렬 D (Trefethen Spectral Methods in MATLAB).

    D[i, j] = c_i (-1)^{i+j} / (c_j (x_i - x_j)),  i≠j.
    D[i, i] = -x_i / 2(1 - x_i²)   (endpoint 제외).
    D[0, 0] = (2N² + 1)/6,  D[N, N] = -(2N² + 1)/6.
    """
    if N < 1:
        raise ValueError("N >= 1 필요")
    x = chebyshev_points(N)
    c = np.array([2.0] + [1.0] * (N - 1) + [2.0]) * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = (c[:, None] / c[None, :]) / (dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D, axis=1))
    return D


def lagrange_interp_1d(
    x_known: NDArray[np.float64],
    y_known: NDArray[np.float64],
    x_new: NDArray[np.float64],
) -> NDArray[np.float64]:
    """1D Lagrange 보간."""
    x_known = np.asarray(x_known, dtype=np.float64)
    y_known = np.asarray(y_known, dtype=np.float64)
    x_new = np.asarray(x_new, dtype=np.float64)
    N = x_known.size
    out = np.zeros_like(x_new)
    for i in range(N):
        L = np.ones_like(x_new)
        for j in range(N):
            if i == j:
                continue
            L *= (x_new - x_known[j]) / (x_known[i] - x_known[j])
        out += y_known[i] * L
    return out


__all__ = ["chebyshev_points", "chebyshev_diff_matrix", "lagrange_interp_1d"]
