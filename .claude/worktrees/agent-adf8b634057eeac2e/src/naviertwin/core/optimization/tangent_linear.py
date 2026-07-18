"""Tangent-linear 방향 도함수 — complex-step / dual-number 방식.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.tangent_linear import directional_derivative
    >>> f = lambda x: float(x[0] ** 2 + x[1] ** 3)
    >>> d = directional_derivative(f, np.array([2.0, 1.0]), np.array([1.0, 0.0]))
    >>> abs(d - 4.0) < 1e-6
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def directional_derivative(
    f: Callable[[NDArray], float], x: NDArray, v: NDArray,
    *, eps: float = 1e-30,
) -> float:
    """complex-step directional derivative: ∂f/∂v ≈ Im(f(x + iεv))/ε."""
    xc = x.astype(np.complex128)
    xc = xc + 1j * eps * v.astype(np.complex128)
    return float(np.imag(f(xc)) / eps)


def jvp_fd(
    f: Callable[[NDArray], NDArray], x: NDArray, v: NDArray,
    *, eps: float = 1e-7,
) -> NDArray[np.float64]:
    """forward-mode Jacobian-vector product via FD."""
    f0 = np.asarray(f(x))
    fp = np.asarray(f(x + eps * v))
    return (fp - f0) / eps


def gradient_from_jvp(
    f: Callable[[NDArray], float], x: NDArray,
    *, eps: float = 1e-30,
) -> NDArray[np.float64]:
    """complex-step gradient (머신 정밀도)."""
    n = x.size
    g = np.zeros(n)
    i = 0
    while i < n:
        v = np.zeros(n)
        v[i] = 1.0
        g[i] = directional_derivative(f, x, v, eps=eps)
        i += 1
    return g


__all__ = ["directional_derivative", "jvp_fd", "gradient_from_jvp"]
