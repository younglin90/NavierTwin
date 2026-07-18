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

from naviertwin._native import _kernels


def clenshaw_curtis_weights(N: int) -> NDArray[np.float64]:
    """Clenshaw-Curtis weights on N+1 Chebyshev-Gauss-Lobatto nodes."""
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by clenshaw_curtis_weights")
    return _kernels.clenshaw_curtis_weights(N)


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
