"""Trust region method — dogleg step.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.trust_region import trust_region_minimize
    >>> def f(x): return float((x - 3.0) ** 2)
    >>> def grad(x): return 2 * (x - 3.0)
    >>> x = trust_region_minimize(f, grad, x0=np.array([0.0]), max_iter=20)
    >>> abs(x[0] - 3.0) < 1e-3
    True
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def trust_region_minimize(
    f: Callable[[NDArray[np.float64]], float],
    grad: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x0: NDArray[np.float64],
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    delta_max: float = 1.0,
) -> NDArray[np.float64]:
    """BFGS Hessian + Cauchy point trust-region (간단 버전)."""
    x = np.asarray(x0, dtype=np.float64).copy()
    n = x.shape[0]
    H = np.eye(n)
    delta = 0.5 * delta_max
    g = grad(x)
    for _ in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        # Cauchy point
        gHg = float(g @ H @ g)
        if gHg <= 0:
            tau = 1.0
        else:
            tau = min(1.0, np.linalg.norm(g) ** 3 / (delta * gHg))
        p = -tau * delta / (np.linalg.norm(g) + 1e-30) * g
        # ratio
        f_x = f(x)
        f_xp = f(x + p)
        m = f_x + g @ p + 0.5 * p @ H @ p
        rho = (f_x - f_xp) / max(f_x - m, 1e-12)
        if rho < 0.25:
            delta *= 0.5
        elif rho > 0.75 and np.linalg.norm(p) >= 0.99 * delta:
            delta = min(2 * delta, delta_max)
        if rho > 0.1:
            x_new = x + p
            g_new = grad(x_new)
            # BFGS update
            s = x_new - x
            y = g_new - g
            sy = float(s @ y)
            if sy > 1e-10:
                rho_b = 1.0 / sy
                Im = np.eye(n)
                H = (Im - rho_b * np.outer(y, s)) @ H @ (Im - rho_b * np.outer(s, y)) \
                    + rho_b * np.outer(y, y)
            x = x_new
            g = g_new
    return x


__all__ = ["trust_region_minimize"]
