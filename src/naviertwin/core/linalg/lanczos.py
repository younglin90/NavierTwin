"""Lanczos iteration — 대칭 행렬용 3항 점화식.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.linalg.lanczos import lanczos
    >>> A = np.diag([5.0, 4.0, 3.0, 2.0, 1.0])
    >>> Q, alpha, beta = lanczos(A, np.ones(5), k=3)
    >>> Q.shape
    (5, 4)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels


def lanczos(
    A: NDArray[np.float64], b: NDArray[np.float64], k: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """k-step Lanczos. 반환: (Q (n, k+1), alpha (k,), beta (k,))."""
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    k = int(min(k, n))
    Q = np.zeros((n, k + 1))
    alpha = np.zeros(k)
    beta = np.zeros(k)
    q = np.asarray(b, dtype=np.float64).ravel()
    q = q / (np.linalg.norm(q) + 1e-30)
    Q[:, 0] = q
    q_prev = np.zeros(n)
    beta_prev = 0.0
    j = 0
    while j < k:
        v = A @ Q[:, j] - beta_prev * q_prev
        a = float(Q[:, j] @ v)
        v = v - a * Q[:, j]
        # full reorthogonalization
        v = v - Q[:, :j + 1] @ (Q[:, :j + 1].T @ v)
        nrm = float(np.linalg.norm(v))
        alpha[j] = a
        beta[j] = nrm
        if nrm < 1e-14:
            return Q[:, :j + 1], alpha[:j + 1], beta[:j + 1]
        q_prev = Q[:, j]
        beta_prev = nrm
        Q[:, j + 1] = v / nrm
        j += 1
    return Q, alpha, beta


def ritz_values_sym(
    A: NDArray[np.float64], b: NDArray[np.float64], k: int,
) -> NDArray[np.float64]:
    """대칭 tridiagonal T 의 eigenvalues = Ritz 추정."""
    _, alpha, beta = lanczos(A, b, k)
    m = alpha.size
    T = np.diag(alpha) + np.diag(beta[:m - 1], 1) + np.diag(beta[:m - 1], -1)
    if _kernels is None:
        raise ImportError("naviertwin._native._kernels is required by ritz_values_sym")
    return _kernels.eigvalsh_symmetric(T)


__all__ = ["lanczos", "ritz_values_sym"]
