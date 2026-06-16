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

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def jacobi(
    A: NDArray[np.float64], b: NDArray[np.float64],
    *, max_iter: int = 1000, tol: float = 1e-8,
    x0: NDArray | None = None,
) -> tuple[NDArray[np.float64], dict]:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    n = b.size
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64).ravel().copy()
    x_native, info = _kernels.jacobi_dense(A, b, x, int(max_iter), float(tol))
    return np.asarray(x_native, dtype=np.float64), dict(info)


def gauss_seidel(
    A: NDArray[np.float64], b: NDArray[np.float64],
    *, max_iter: int = 1000, tol: float = 1e-8,
    x0: NDArray | None = None,
) -> tuple[NDArray[np.float64], dict]:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    n = b.size
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64).ravel().copy()
    x_native, info = _kernels.gauss_seidel_dense(A, b, x, int(max_iter), float(tol))
    return np.asarray(x_native, dtype=np.float64), dict(info)


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
    mi = max_iter if max_iter is not None else 2 * n
    x_native, info = _kernels.conjugate_gradient_dense(A, b, x, int(mi), float(tol))
    return np.asarray(x_native, dtype=np.float64), dict(info)


__all__ = ["jacobi", "gauss_seidel", "conjugate_gradient"]
