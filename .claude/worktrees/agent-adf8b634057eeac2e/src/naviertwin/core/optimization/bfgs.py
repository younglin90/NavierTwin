"""BFGS quasi-Newton — Hessian inverse 근사 rank-2 update + Armijo line search.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.bfgs import bfgs_minimize
    >>> def f(x): return float(x @ x + 0.5 * x[0] ** 4)
    >>> def g(x): return 2 * x + np.array([2 * x[0] ** 3, 0])
    >>> x, info = bfgs_minimize(f, g, np.array([3.0, -2.0]))
    >>> info["converged"]
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.core.optimization.line_search import armijo_backtrack

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required by BFGS optimization")


def bfgs_minimize(
    f: Callable[[NDArray], float],
    grad: Callable[[NDArray], NDArray],
    x0: NDArray[np.float64],
    *, max_iter: int = 200, tol: float = 1e-8,
) -> tuple[NDArray[np.float64], dict]:
    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    n = x.size
    H = np.eye(n)
    g = grad(x)
    i = 0
    while i < max_iter:
        gnorm = float(_kernels.vector_l2_norm(np.asarray(g, dtype=np.float64)))
        if gnorm < tol:
            return x, {"iters": i, "grad_norm": gnorm, "converged": True, "f": float(f(x))}
        p = -H @ g
        # descent 방향 확인
        if _kernels.vector_dot(np.asarray(p, dtype=np.float64), np.asarray(g, dtype=np.float64)) > 0:
            H = np.eye(n)
            p = -g
        alpha = armijo_backtrack(f, grad, x, p, alpha0=1.0)
        s = alpha * p
        x_new = x + s
        g_new = grad(x_new)
        y = g_new - g
        sy = float(_kernels.vector_dot(np.asarray(s, dtype=np.float64), np.asarray(y, dtype=np.float64)))
        if sy > 1e-12:
            rho = 1.0 / sy
            eye = np.eye(n)
            Vt = eye - rho * np.outer(y, s)
            V = eye - rho * np.outer(s, y)
            H = V @ H @ Vt + rho * np.outer(s, s)
        x = x_new
        g = g_new
        i += 1
    return x, {
        "iters": max_iter, "grad_norm": float(_kernels.vector_l2_norm(np.asarray(g, dtype=np.float64))),
        "converged": False, "f": float(f(x)),
    }


__all__ = ["bfgs_minimize"]
