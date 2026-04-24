"""Line search — Armijo backtracking + Wolfe 조건 체크.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.line_search import armijo_backtrack
    >>> def f(x): return float(x @ x)
    >>> def g(x): return 2 * x
    >>> x = np.array([3.0, -2.0])
    >>> p = -g(x)
    >>> alpha = armijo_backtrack(f, g, x, p)
    >>> alpha > 0 and f(x + alpha * p) < f(x)
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def armijo_backtrack(
    f: Callable[[NDArray], float],
    grad: Callable[[NDArray], NDArray],
    x: NDArray[np.float64],
    p: NDArray[np.float64],
    *,
    alpha0: float = 1.0, c1: float = 1e-4, rho: float = 0.5,
    max_iter: int = 50,
) -> float:
    """Armijo back-tracking: f(x+αp) ≤ f(x) + c1 α gᵀp."""
    alpha = float(alpha0)
    f0 = f(x)
    g0 = grad(x)
    slope = float(g0 @ p)
    for _ in range(max_iter):
        if f(x + alpha * p) <= f0 + c1 * alpha * slope:
            return alpha
        alpha *= rho
    return alpha


def check_wolfe(
    f: Callable[[NDArray], float], grad: Callable[[NDArray], NDArray],
    x: NDArray, p: NDArray, alpha: float,
    *, c1: float = 1e-4, c2: float = 0.9,
) -> dict[str, bool]:
    """(Strong) Wolfe 조건 두 개 평가."""
    f0 = f(x)
    g0 = grad(x)
    slope0 = float(g0 @ p)
    f1 = f(x + alpha * p)
    g1 = grad(x + alpha * p)
    slope1 = float(g1 @ p)
    armijo = f1 <= f0 + c1 * alpha * slope0
    curvature = slope1 >= c2 * slope0  # 일반 Wolfe
    strong = abs(slope1) <= c2 * abs(slope0)
    return {"armijo": armijo, "curvature": curvature, "strong_wolfe": armijo and strong}


__all__ = ["armijo_backtrack", "check_wolfe"]
