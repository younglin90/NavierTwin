"""Clenshaw-Curtis quadrature — Chebyshev nodes 에서 ∫_{-1}^{1} f(x) dx.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.numerics.chebyshev import chebyshev_points
    >>> from naviertwin.core.numerics.clenshaw_curtis import (
    ...     clenshaw_curtis_weights,
    ...     integrate_cc,
    ... )
    >>> N = 20
    >>> x = chebyshev_points(N)
    >>> val = integrate_cc(np.cos(x), N)
    >>> abs(val - 2 * np.sin(1.0)) < 1e-10
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def clenshaw_curtis_weights(N: int) -> NDArray[np.float64]:
    """Clenshaw-Curtis weights on N+1 Chebyshev-Gauss-Lobatto nodes."""
    if N < 2:
        raise ValueError("N >= 2")
    theta = np.pi * np.arange(N + 1) / N
    w = np.zeros(N + 1)
    for j in range(N + 1):
        s = 0.0
        for k in range(1, N // 2 + 1):
            denom = (4 * k * k - 1)
            s += (2.0 / denom) * np.cos(2 * k * theta[j])
        b = 1.0 if (N % 2 == 1) else 1.0 - np.cos(N * theta[j]) / (N * N - 1 + 1e-30)
        w[j] = (2.0 / N) * (1.0 - s - b * 0.0)  # approximation; use explicit form below

    # use standard explicit form (Trefethen):
    w = np.zeros(N + 1)
    w[0] = 1.0 / (N * N - 1 + (N % 2))
    w[N] = w[0]
    for j in range(1, N):
        s = 0.0
        for k in range(1, N // 2 + 1):
            b = 1.0 if (2 * k < N) else 0.5
            s += b * np.cos(2 * k * theta[j]) / (4 * k * k - 1)
        w[j] = (2.0 / N) * (1.0 - 2.0 * s)
    return w


def integrate_cc(f_values: NDArray[np.float64], N: int) -> float:
    """f 값 (N+1 노드에서) → ∫ f dx on [-1,1]."""
    w = clenshaw_curtis_weights(N)
    return float(w @ f_values)


def integrate_cc_interval(
    f_values: NDArray[np.float64], N: int, a: float, b: float,
) -> float:
    """[a, b] 적분 (선형 변환)."""
    return 0.5 * (b - a) * integrate_cc(f_values, N)


__all__ = ["clenshaw_curtis_weights", "integrate_cc", "integrate_cc_interval"]
