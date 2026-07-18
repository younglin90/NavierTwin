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

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


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
    return np.asarray(
        _kernels.thomas_solve(
            np.asarray(a, dtype=np.float64),
            np.asarray(b, dtype=np.float64),
            np.asarray(c, dtype=np.float64),
            np.asarray(d, dtype=np.float64),
        ),
        dtype=np.float64,
    )


def crank_nicolson_heat_step(
    u: NDArray[np.float64], nu: float, dx: float, dt: float,
) -> NDArray[np.float64]:
    """Crank-Nicolson heat 1D step (Dirichlet)."""
    n = u.size
    r = nu * dt / (2 * dx ** 2)
    a = np.full(n, -r)
    b = np.full(n, 1 + 2 * r)
    c = np.full(n, -r)
    a[0] = 0.0
    c[-1] = 0.0
    d = u.copy()
    d[1:-1] = (1 - 2 * r) * u[1:-1] + r * (u[2:] + u[:-2])
    d[0] = 0.0
    d[-1] = 0.0
    b[0] = 1.0
    c[0] = 0.0
    b[-1] = 1.0
    a[-1] = 0.0
    return thomas_solve(a, b, c, d)


__all__ = ["thomas_solve", "crank_nicolson_heat_step"]
