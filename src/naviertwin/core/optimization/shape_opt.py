"""Shape optimization — parametric Bezier control points.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.shape_opt import bezier_eval, optimize_bezier
    >>> ctrl = np.array([[0., 0], [0.5, 1], [1., 0]])
    >>> pts = bezier_eval(ctrl, t=np.linspace(0, 1, 5))
    >>> pts.shape
    (5, 2)
"""

from __future__ import annotations

from collections.abc import Callable
from math import comb

import numpy as np
from numpy.typing import NDArray


def bezier_eval(
    ctrl: NDArray[np.float64], t: NDArray[np.float64],
) -> NDArray[np.float64]:
    """De Casteljau Bezier 평가."""
    ctrl = np.asarray(ctrl, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    n = ctrl.shape[0] - 1
    powers = np.arange(n + 1)
    coeffs = np.fromiter(map(lambda i: comb(n, int(i)), powers), dtype=np.float64)
    basis = coeffs[None, :] * (1 - t[:, None]) ** (n - powers) * t[:, None] ** powers
    return basis @ ctrl


def optimize_bezier(
    objective: Callable[[NDArray], float],
    ctrl0: NDArray[np.float64],
    *,
    fixed_endpoints: bool = True,
    max_iter: int = 50,
    lr: float = 0.05,
    eps: float = 1e-4,
) -> NDArray[np.float64]:
    """Bezier 제어점에 대해 finite-diff gradient descent."""
    ctrl = np.asarray(ctrl0, dtype=np.float64).copy()
    iter_idx = 0
    while iter_idx < max_iter:
        g = np.zeros_like(ctrl)
        f0 = objective(ctrl)
        row_start = 1 if fixed_endpoints else 0
        row_stop = len(ctrl) - 1 if fixed_endpoints else len(ctrl)
        rows, cols = np.indices((row_stop - row_start, ctrl.shape[1]))
        rows = rows.ravel() + row_start
        cols = cols.ravel()
        trials = np.repeat(ctrl[None, :, :], rows.size, axis=0)
        trials[np.arange(rows.size), rows, cols] += eps
        values = np.fromiter(map(objective, trials), dtype=np.float64, count=rows.size)
        g[rows, cols] = (values - f0) / eps
        ctrl = ctrl - lr * g
        iter_idx += 1
    return ctrl


__all__ = ["bezier_eval", "optimize_bezier"]
