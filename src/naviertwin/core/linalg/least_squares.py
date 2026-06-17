"""비선형 최소자승 — Gauss-Newton / Levenberg-Marquardt.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.linalg.least_squares import levenberg_marquardt
    >>> t = np.linspace(0, 1, 20)
    >>> y = 2.0 * np.exp(-3.0 * t) + 0.01 * np.random.default_rng(0).standard_normal(20)
    >>> def r(p): return p[0] * np.exp(-p[1] * t) - y
    >>> p, info = levenberg_marquardt(r, p0=np.array([1.0, 1.0]))
    >>> abs(p[0] - 2.0) < 0.1 and abs(p[1] - 3.0) < 0.2
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

R_T = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def _fd_jacobian(r: R_T, p: NDArray[np.float64], eps: float = 1e-6) -> NDArray[np.float64]:
    r0 = r(p)
    m = r0.size
    n = p.size
    J = np.zeros((m, n))
    j = 0
    while j < n:
        pp = p.copy()
        pp[j] += eps
        J[:, j] = (r(pp) - r0) / eps
        j += 1
    return J


def levenberg_marquardt(
    residual: R_T, p0: NDArray[np.float64],
    *,
    jac: Callable | None = None,
    max_iter: int = 100, tol: float = 1e-10,
    lam0: float = 1e-3, lam_up: float = 10.0, lam_down: float = 10.0,
) -> tuple[NDArray[np.float64], dict]:
    p = np.asarray(p0, dtype=np.float64).ravel().copy()
    lam = float(lam0)
    i = 0
    while i < max_iter:
        r = residual(p)
        cost = 0.5 * r @ r
        if cost < tol:
            return p, {"iters": i, "cost": float(cost), "converged": True, "lambda": lam}
        J = jac(p) if jac else _fd_jacobian(residual, p)
        JTJ = J.T @ J
        g = J.T @ r
        step_ok = False
        trial = 0
        while trial < 20:
            A = JTJ + lam * np.eye(JTJ.shape[0])
            try:
                if _kernels is None:
                    raise RuntimeError("naviertwin._native._kernels is required")
                dp = _kernels.solve_dense(A, -g)
            except Exception:
                dp = np.linalg.lstsq(A, -g, rcond=None)[0]
            p_trial = p + dp
            r_trial = residual(p_trial)
            cost_trial = 0.5 * r_trial @ r_trial
            if cost_trial < cost:
                p = p_trial
                lam = max(lam / lam_down, 1e-12)
                step_ok = True
                break
            lam = min(lam * lam_up, 1e12)
            trial += 1
        if not step_ok:
            return p, {
                "iters": i, "cost": float(cost), "converged": False, "lambda": lam,
            }
        i += 1
    return p, {
        "iters": max_iter, "cost": 0.5 * float(np.linalg.norm(residual(p)) ** 2),
        "converged": False, "lambda": lam,
    }


def gauss_newton(
    residual: R_T, p0: NDArray[np.float64],
    *, jac: Callable | None = None,
    max_iter: int = 50, tol: float = 1e-10,
) -> tuple[NDArray[np.float64], dict]:
    p = np.asarray(p0, dtype=np.float64).ravel().copy()
    i = 0
    while i < max_iter:
        r = residual(p)
        cost = 0.5 * r @ r
        if cost < tol:
            return p, {"iters": i, "cost": float(cost), "converged": True}
        J = jac(p) if jac else _fd_jacobian(residual, p)
        dp, *_ = np.linalg.lstsq(J, -r, rcond=None)
        p = p + dp
        i += 1
    return p, {
        "iters": max_iter,
        "cost": 0.5 * float(np.linalg.norm(residual(p)) ** 2),
        "converged": False,
    }


__all__ = ["levenberg_marquardt", "gauss_newton"]
