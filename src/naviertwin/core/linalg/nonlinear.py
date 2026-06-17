"""비선형 방정식 근 찾기 — Newton / damped Newton / Broyden.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.linalg.nonlinear import newton_solve
    >>> def F(x): return np.array([x[0]**2 - 4.0])
    >>> def J(x): return np.array([[2*x[0]]])
    >>> x, info = newton_solve(F, J, x0=np.array([1.0]))
    >>> abs(x[0] - 2.0) < 1e-8
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

F_T = Callable[[NDArray[np.float64]], NDArray[np.float64]]
J_T = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def newton_solve(
    F: F_T, J: J_T, x0: NDArray[np.float64],
    *, tol: float = 1e-10, max_iter: int = 50,
    damping: float = 1.0,
) -> tuple[NDArray[np.float64], dict]:
    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    i = 0
    while i < max_iter:
        fx = F(x)
        if np.linalg.norm(fx) < tol:
            return x, {"iters": i, "residual": float(np.linalg.norm(fx)), "converged": True}
        Jx = J(x)
        try:
            if _kernels is None:
                raise RuntimeError("naviertwin._native._kernels is required")
            dx = _kernels.solve_dense(Jx, -fx)
        except Exception:
            dx = np.linalg.lstsq(Jx, -fx, rcond=None)[0]
        x = x + damping * dx
        i += 1
    return x, {
        "iters": max_iter,
        "residual": float(np.linalg.norm(F(x))),
        "converged": False,
    }


def broyden_solve(
    F: F_T, x0: NDArray[np.float64],
    *, tol: float = 1e-10, max_iter: int = 100,
) -> tuple[NDArray[np.float64], dict]:
    """Jacobian-free Broyden (rank-1 update)."""
    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    fx = F(x)
    n = x.size
    B_inv = np.eye(n)
    i = 0
    while i < max_iter:
        if np.linalg.norm(fx) < tol:
            return x, {"iters": i, "residual": float(np.linalg.norm(fx)), "converged": True}
        dx = -B_inv @ fx
        x_new = x + dx
        fx_new = F(x_new)
        df = fx_new - fx
        denom = dx @ (B_inv @ df)
        if abs(denom) > 1e-30:
            B_inv = B_inv + np.outer(dx - B_inv @ df, dx @ B_inv) / denom
        x = x_new
        fx = fx_new
        i += 1
    return x, {
        "iters": max_iter,
        "residual": float(np.linalg.norm(fx)),
        "converged": False,
    }


def fd_jacobian(F: F_T, x: NDArray[np.float64], eps: float = 1e-6) -> NDArray[np.float64]:
    """1차 FD Jacobian — J 를 수동으로 쓰지 않을 때."""
    x = np.asarray(x, dtype=np.float64).ravel()
    f0 = F(x)
    m = f0.size
    n = x.size
    J = np.zeros((m, n))
    j = 0
    while j < n:
        xp = x.copy()
        xp[j] += eps
        J[:, j] = (F(xp) - f0) / eps
        j += 1
    return J


__all__ = ["newton_solve", "broyden_solve", "fd_jacobian"]
