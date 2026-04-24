"""Sequential Quadratic Programming — equality constraint version (간단).

min f(x) s.t. h(x) = 0.  KKT 시스템 lin. solve.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.sqp import sqp_eq
    >>> # min x²+y² s.t. x+y=1 → (0.5, 0.5)
    >>> def f(x): return float(x @ x)
    >>> def grad(x): return 2 * x
    >>> def hess(x): return 2 * np.eye(2)
    >>> def h(x): return np.array([x[0] + x[1] - 1.0])
    >>> def hjac(x): return np.array([[1.0, 1.0]])
    >>> x = sqp_eq(f, grad, hess, h, hjac, x0=np.zeros(2), max_iter=20)
    >>> np.allclose(x, [0.5, 0.5], atol=1e-4)
    True
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def sqp_eq(
    f: Callable[[NDArray], float],
    grad: Callable[[NDArray], NDArray],
    hess: Callable[[NDArray], NDArray],
    h: Callable[[NDArray], NDArray],
    hjac: Callable[[NDArray], NDArray],
    x0: NDArray[np.float64],
    *,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> NDArray[np.float64]:
    """KKT step: [B Aᵀ; A 0][p; λ] = [-g; -h]."""
    x = np.asarray(x0, dtype=np.float64).copy()
    for _ in range(max_iter):
        g = grad(x)
        B = hess(x)
        c = h(x)
        A = hjac(x)
        n = len(x)
        m = len(c)
        K = np.block([[B, A.T], [A, np.zeros((m, m))]])
        rhs = np.concatenate([-g, -c])
        sol = np.linalg.solve(K, rhs)
        p = sol[:n]
        x = x + p
        if np.linalg.norm(p) < tol:
            break
    return x


__all__ = ["sqp_eq"]
