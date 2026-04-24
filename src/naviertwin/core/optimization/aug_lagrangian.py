"""Augmented Lagrangian — equality constraints.

L_A(x, λ, μ) = f + λ·h + (μ/2)‖h‖².

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.aug_lagrangian import aug_lagrangian
    >>> def f(x): return float(x @ x)
    >>> def grad(x): return 2 * x
    >>> def h(x): return np.array([x[0] + x[1] - 1.0])
    >>> def hjac(x): return np.array([[1.0, 1.0]])
    >>> x = aug_lagrangian(f, grad, h, hjac, x0=np.zeros(2))
    >>> np.allclose(x, [0.5, 0.5], atol=1e-3)
    True
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def _grad_descent(
    g_fn: Callable[[NDArray], NDArray], x0: NDArray, *, lr: float = 0.05,
    n: int = 200, tol: float = 1e-8,
) -> NDArray:
    x = x0.copy()
    for _ in range(n):
        gx = g_fn(x)
        x = x - lr * gx
        if np.linalg.norm(gx) < tol:
            break
    return x


def aug_lagrangian(
    f: Callable[[NDArray], float],
    grad: Callable[[NDArray], NDArray],
    h: Callable[[NDArray], NDArray],
    hjac: Callable[[NDArray], NDArray],
    x0: NDArray[np.float64],
    *,
    mu: float = 10.0,
    n_outer: int = 20,
) -> NDArray[np.float64]:
    """outer iteration: λ_{k+1} = λ_k + μ h(x_k)."""
    x = np.asarray(x0, dtype=np.float64).copy()
    lam = np.zeros(h(x).size)
    for _ in range(n_outer):
        # solve unconstrained augmented problem
        def g_aug(x_):
            return grad(x_) + hjac(x_).T @ (lam + mu * h(x_))
        x = _grad_descent(g_aug, x, lr=1.0 / max(mu, 1.0), n=300)
        lam = lam + mu * h(x)
        mu *= 1.5
    return x


__all__ = ["aug_lagrangian"]
