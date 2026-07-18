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

from naviertwin._native import _kernels


def chebyshev_points(N: int) -> NDArray[np.float64]:
    """Chebyshev-Gauss-Lobatto 점들 x_k = cos(kπ/N), k=0..N.

    Returns:
        (N+1,) [1, ..., -1].
    """
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by chebyshev_points")
    return _kernels.chebyshev_points(N)


def chebyshev_diff_matrix(N: int) -> NDArray[np.float64]:
    """스펙트럴 미분 행렬 D (Trefethen Spectral Methods in MATLAB).

    D[i, j] = c_i (-1)^{i+j} / (c_j (x_i - x_j)),  i≠j.
    D[i, i] = -x_i / 2(1 - x_i²)   (endpoint 제외).
    D[0, 0] = (2N² + 1)/6,  D[N, N] = -(2N² + 1)/6.
    """
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by chebyshev_diff_matrix")
    return _kernels.chebyshev_diff_matrix(N)


def lagrange_interp_1d(
    x_known: NDArray[np.float64],
    y_known: NDArray[np.float64],
    x_new: NDArray[np.float64],
) -> NDArray[np.float64]:
    """1D Lagrange 보간."""
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by lagrange_interp_1d")
    return _kernels.lagrange_interp_1d(x_known, y_known, x_new)


__all__ = ["chebyshev_points", "chebyshev_diff_matrix", "lagrange_interp_1d"]
