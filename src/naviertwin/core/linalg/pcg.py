"""Preconditioned Conjugate Gradient (Jacobi preconditioner).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.linalg.pcg import pcg
    >>> A = np.array([[4., 1.], [1., 3.]])
    >>> b = np.array([1., 2.])
    >>> x, info = pcg(A, b)
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


def jacobi_preconditioner(A: NDArray[np.float64]) -> Callable:
    d = np.diag(A)
    if np.any(d == 0):
        raise ValueError("zero diagonal")
    inv_d = 1.0 / d
    return lambda r: inv_d * r


def pcg(
    A: NDArray[np.float64], b: NDArray[np.float64],
    *, M: Callable | None = None,
    max_iter: int | None = None, tol: float = 1e-10,
    x0: NDArray | None = None,
) -> tuple[NDArray[np.float64], dict]:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    n = b.size
    x = np.zeros(n) if x0 is None else np.asarray(x0).ravel().copy()
    mi = max_iter or 2 * n
    if M is None:
        x_native, info = _kernels.pcg_jacobi(A, b, x, int(mi), float(tol))
        return np.asarray(x_native, dtype=np.float64), dict(info)
    return _pcg_with_preconditioner(A, b, x, M, mi, tol)


def _pcg_with_preconditioner(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    x: NDArray[np.float64],
    M: Callable,
    max_iter: int,
    tol: float,
) -> tuple[NDArray[np.float64], dict]:
    r = b - A @ x
    z = M(r)
    p = z.copy()
    rz = float(r @ z)
    i = 0
    while i < max_iter:
        Ap = A @ p
        alpha = rz / (p @ Ap + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        if np.linalg.norm(r) < tol:
            return x, {"iters": i + 1, "residual": float(np.linalg.norm(r)),
                       "converged": True}
        z = M(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
        i += 1
    return x, {"iters": max_iter, "residual": float(np.linalg.norm(r)), "converged": False}


__all__ = ["pcg", "jacobi_preconditioner"]
