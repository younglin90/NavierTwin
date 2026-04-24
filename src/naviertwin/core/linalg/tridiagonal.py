"""Tridiagonal 선형시스템 — Thomas 알고리즘 (O(n)).

a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i, 양 끝은 a_0=c_{n-1}=0.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.linalg.tridiagonal import thomas_solve
    >>> a = np.array([0., 1., 1.])
    >>> b = np.array([2., 2., 2.])
    >>> c = np.array([1., 1., 0.])
    >>> d = np.array([3., 4., 3.])
    >>> x = thomas_solve(a, b, c, d)
    >>> np.allclose(np.array([[2,1,0],[1,2,1],[0,1,2]]) @ x, d)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def thomas_solve(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
    d: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Thomas algorithm (비파괴적)."""
    n = len(d)
    if not (len(a) == len(b) == len(c) == n):
        raise ValueError("diagonal 길이 불일치")
    a = np.asarray(a, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).copy()
    c = np.asarray(c, dtype=np.float64).copy()
    d = np.asarray(d, dtype=np.float64).copy()
    # forward
    for i in range(1, n):
        m = a[i] / b[i - 1]
        b[i] = b[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]
    # back substitution
    x = np.zeros(n, dtype=np.float64)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def crank_nicolson_heat_step(
    u: NDArray[np.float64], nu: float, dx: float, dt: float,
) -> NDArray[np.float64]:
    """1 step of Crank-Nicolson for heat 1D (Dirichlet)."""
    n = u.size
    r = nu * dt / (2 * dx ** 2)
    a = np.full(n, -r)
    b = np.full(n, 1 + 2 * r)
    c = np.full(n, -r)
    a[0] = 0.0
    c[-1] = 0.0
    # RHS
    d = u.copy()
    d[1:-1] = (1 - 2 * r) * u[1:-1] + r * (u[2:] + u[:-2])
    # Dirichlet 0
    d[0] = 0.0
    d[-1] = 0.0
    b[0] = 1.0
    c[0] = 0.0
    b[-1] = 1.0
    a[-1] = 0.0
    return thomas_solve(a, b, c, d)


__all__ = ["thomas_solve", "crank_nicolson_heat_step"]
