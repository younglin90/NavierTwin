"""Counterfactual explanation — minimal feature change s.t. y_target.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.twin.counterfactual import minimal_change
    >>> def f(x): return x[0] + 0.5
    >>> dx = minimal_change(np.array([0.0]), f, target=2.0)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def minimal_change(
    x0: NDArray[np.float64],
    f: Callable[[NDArray], float],
    *,
    target: float, lr: float = 0.05, n_iter: int = 200,
    eps: float = 1e-4, lam: float = 0.1,
) -> NDArray[np.float64]:
    """Minimize (f(x)-target)² + lam ‖x-x0‖² via gradient descent (FD)."""
    x0 = np.asarray(x0, dtype=np.float64)
    x = x0.copy()
    for _ in range(n_iter):
        # FD gradient of (f(x)-target)²
        g = np.zeros_like(x)
        f_now = f(x) - target
        for i in range(len(x)):
            xi = x.copy()
            xi[i] += eps
            g[i] = ((f(xi) - target) - f_now) / eps * 2 * f_now
        g += 2 * lam * (x - x0)
        x = x - lr * g
    return x - x0  # delta


__all__ = ["minimal_change"]
