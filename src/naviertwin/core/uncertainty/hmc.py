"""Hamiltonian Monte Carlo — leapfrog + Metropolis accept.

FD gradient 사용 (사용자가 analytic grad 제공하면 더 정확).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.uncertainty.hmc import hmc
    >>> logp = lambda q: -0.5 * float(q @ q)
    >>> samples = hmc(logp, np.zeros(1), n=500, step=0.1, L=20, seed=0)
    >>> abs(samples.mean()) < 0.2
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def _fd_grad(
    f: Callable[[NDArray], float], x: NDArray, eps: float = 1e-5,
) -> NDArray:
    g = np.zeros_like(x)
    f0 = f(x)
    for i in range(x.size):
        xp = x.copy()
        xp[i] += eps
        g[i] = (f(xp) - f0) / eps
    return g


def hmc(
    log_prob: Callable[[NDArray[np.float64]], float],
    q0: NDArray[np.float64],
    *, n: int = 1000, step: float = 0.1, L: int = 10,
    grad: Callable | None = None,
    seed: int | None = 0,
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    q = np.asarray(q0, dtype=np.float64).ravel().copy()
    d = q.size
    out = np.zeros((n, d))
    grad = grad if grad is not None else (lambda x: _fd_grad(log_prob, x))

    for i in range(n):
        p = rng.standard_normal(d)
        q_new = q.copy()
        p_new = p.copy()
        # leapfrog
        p_new = p_new + 0.5 * step * grad(q_new)
        for step_i in range(L):
            q_new = q_new + step * p_new
            if step_i < L - 1:
                p_new = p_new + step * grad(q_new)
        p_new = p_new + 0.5 * step * grad(q_new)
        p_new = -p_new

        current_H = -log_prob(q) + 0.5 * (p @ p)
        new_H = -log_prob(q_new) + 0.5 * (p_new @ p_new)
        if np.log(rng.random()) < (current_H - new_H):
            q = q_new
        out[i] = q
    return out


__all__ = ["hmc"]
