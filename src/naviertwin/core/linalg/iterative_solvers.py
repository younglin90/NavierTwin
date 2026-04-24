"""선형 시스템 반복 솔버 — Jacobi / Gauss-Seidel / CG.

CFD 압력 포아송 / ROM 잔차 보정 등에 사용.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.linalg.iterative_solvers import conjugate_gradient
    >>> A = np.array([[4., 1.], [1., 3.]])
    >>> b = np.array([1., 2.])
    >>> x, info = conjugate_gradient(A, b)
    >>> np.allclose(A @ x, b, atol=1e-8)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def jacobi(
    A: NDArray[np.float64], b: NDArray[np.float64],
    *, max_iter: int = 1000, tol: float = 1e-8,
    x0: NDArray | None = None,
) -> tuple[NDArray[np.float64], dict]:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    n = b.size
    D = np.diag(A).copy()
    if np.any(D == 0):
        raise ValueError("zero diagonal")
    R = A - np.diag(D)
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64).ravel().copy()
    for i in range(max_iter):
        x_new = (b - R @ x) / D
        err = np.linalg.norm(A @ x_new - b)
        x = x_new
        if err < tol:
            return x, {"iters": i + 1, "residual": float(err), "converged": True}
    return x, {"iters": max_iter, "residual": float(err), "converged": False}


def gauss_seidel(
    A: NDArray[np.float64], b: NDArray[np.float64],
    *, max_iter: int = 1000, tol: float = 1e-8,
    x0: NDArray | None = None,
) -> tuple[NDArray[np.float64], dict]:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    n = b.size
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64).ravel().copy()
    for i in range(max_iter):
        for k in range(n):
            s = b[k] - A[k, :k] @ x[:k] - A[k, k + 1:] @ x[k + 1:]
            x[k] = s / A[k, k]
        err = np.linalg.norm(A @ x - b)
        if err < tol:
            return x, {"iters": i + 1, "residual": float(err), "converged": True}
    return x, {"iters": max_iter, "residual": float(err), "converged": False}


def conjugate_gradient(
    A: NDArray[np.float64], b: NDArray[np.float64],
    *, max_iter: int | None = None, tol: float = 1e-10,
    x0: NDArray | None = None,
) -> tuple[NDArray[np.float64], dict]:
    """SPD 시스템용 CG."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    n = b.size
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64).ravel().copy()
    r = b - A @ x
    p = r.copy()
    rs_old = r @ r
    mi = max_iter if max_iter is not None else 2 * n
    for i in range(mi):
        Ap = A @ p
        alpha = rs_old / (p @ Ap + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        if np.sqrt(rs_new) < tol:
            return x, {"iters": i + 1, "residual": float(np.sqrt(rs_new)), "converged": True}
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x, {"iters": mi, "residual": float(np.sqrt(rs_new)), "converged": False}


__all__ = ["jacobi", "gauss_seidel", "conjugate_gradient"]
