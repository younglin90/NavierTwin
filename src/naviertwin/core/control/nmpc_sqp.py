"""Nonlinear MPC - finite-horizon optimization via gradient descent on u sequence.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.control.nmpc_sqp import nmpc_solve
    >>> def f(x, u): return x + u
    >>> def cost(x, u): return float(x**2 + 0.1*u**2)
    >>> u_seq = nmpc_solve(f, cost, x0=np.array([1.0]), N=10)
    >>> u_seq.shape[0]
    10
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def nmpc_solve(
    dynamics: Callable[[NDArray, NDArray], NDArray],
    stage_cost: Callable[[NDArray, NDArray], float],
    x0: NDArray[np.float64],
    *,
    N: int = 10,
    n_u: int = 1,
    n_iter: int = 200,
    lr: float = 0.05,
    eps: float = 1e-4,
) -> NDArray[np.float64]:
    """gradient descent (FD) on flat u sequence (N, n_u)."""
    x0 = np.asarray(x0, dtype=np.float64)
    u_seq = np.zeros((N, n_u))

    def total_cost(us):
        x = x0.copy()
        c = 0.0
        k = 0
        while k < N:
            c += stage_cost(x, us[k])
            x = dynamics(x, us[k])
            k += 1
        return c

    flat_u = u_seq.ravel()
    g = np.zeros_like(u_seq)
    flat_g = g.ravel()
    it = 0
    while it < n_iter:
        c0 = total_cost(u_seq)
        flat_g.fill(0.0)
        idx = 0
        while idx < flat_u.size:
            flat_u[idx] += eps
            flat_g[idx] = (total_cost(u_seq) - c0) / eps
            flat_u[idx] -= eps
            idx += 1
        u_seq -= lr * g
        it += 1
    return u_seq


__all__ = ["nmpc_solve"]
