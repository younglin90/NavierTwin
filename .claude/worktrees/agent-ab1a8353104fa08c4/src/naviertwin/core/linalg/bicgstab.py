"""BiCGStab — 비대칭 선형시스템 풀이 (van der Vorst 1992).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.linalg.bicgstab import bicgstab
    >>> A = np.array([[4., 1., 0.], [2., 5., 1.], [0., 1., 3.]])
    >>> b = np.array([1., 2., 3.])
    >>> x, info = bicgstab(A, b)
    >>> info["converged"]
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def bicgstab(
    A, b: NDArray[np.float64],
    *, x0: NDArray | None = None, max_iter: int | None = None,
    tol: float = 1e-10, M: Callable | None = None,
) -> tuple[NDArray[np.float64], dict]:
    """A 는 dense ndarray 또는 linear op callable."""
    b = np.asarray(b, dtype=np.float64).ravel()
    n = b.size
    x = np.zeros(n) if x0 is None else np.asarray(x0).ravel().copy()
    mi = max_iter or 2 * n
    if not callable(A) and M is None:
        x_native, info = _kernels.bicgstab_dense(np.asarray(A, dtype=np.float64), b, x, int(mi), float(tol))
        return np.asarray(x_native, dtype=np.float64), dict(info)
    return _bicgstab_with_operator(A, b, x, mi, tol, M)


def _bicgstab_with_operator(
    A,
    b: NDArray[np.float64],
    x: NDArray[np.float64],
    max_iter: int,
    tol: float,
    M: Callable | None,
) -> tuple[NDArray[np.float64], dict]:
    n = b.size
    if callable(A):
        A_fn = A
    else:
        A_arr = np.asarray(A, dtype=np.float64)

        def A_fn(v):
            return A_arr @ v

    M_fn = M if M is not None else (lambda v: v)
    r = b - A_fn(x)
    r_hat = r.copy()
    rho_prev = alpha = omega = 1.0
    v = np.zeros(n)
    p = np.zeros(n)
    i = 0
    while i < max_iter:
        rho = float(r_hat @ r)
        if abs(rho) < 1e-30:
            break
        beta = (rho / rho_prev) * (alpha / (omega + 1e-30))
        p = r + beta * (p - omega * v)
        y = M_fn(p)
        v = A_fn(y)
        alpha = rho / (r_hat @ v + 1e-30)
        s = r - alpha * v
        if np.linalg.norm(s) < tol:
            x = x + alpha * y
            return x, {"iters": i + 1, "residual": float(np.linalg.norm(b - A_fn(x))),
                       "converged": True}
        z = M_fn(s)
        t = A_fn(z)
        omega = float(t @ s) / (t @ t + 1e-30)
        x = x + alpha * y + omega * z
        r = s - omega * t
        if np.linalg.norm(r) < tol:
            return x, {"iters": i + 1, "residual": float(np.linalg.norm(r)),
                       "converged": True}
        rho_prev = rho
        i += 1
    return x, {"iters": max_iter, "residual": float(np.linalg.norm(b - A_fn(x))),
               "converged": False}


__all__ = ["bicgstab"]
