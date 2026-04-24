"""SVD 유틸 — thin / truncated / randomized / low-rank approx.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.linalg.svd_utils import truncated_svd
    >>> A = np.random.randn(50, 20)
    >>> U, s, Vt = truncated_svd(A, k=5)
    >>> U.shape, s.shape, Vt.shape
    ((50, 5), (5,), (5, 20))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def truncated_svd(
    A: NDArray[np.float64], k: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """thin SVD 의 상위 k 모드."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k], s[:k], Vt[:k, :]


def randomized_svd(
    A: NDArray[np.float64], k: int, *,
    n_oversamples: int = 10, n_iter: int = 2, seed: int | None = 0,
) -> tuple[NDArray, NDArray, NDArray]:
    """Halko 2011 randomized SVD — 대규모 행렬에 훨씬 빠름."""
    rng = np.random.default_rng(seed)
    m, n = A.shape
    r = min(k + n_oversamples, n)
    Omega = rng.standard_normal((n, r))
    Y = A @ Omega
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(Y)
        Z = A.T @ Q
        Q2, _ = np.linalg.qr(Z)
        Y = A @ Q2
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ A  # r × n
    U_tilde, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde
    return U[:, :k], s[:k], Vt[:k, :]


def low_rank_reconstruct(
    U: NDArray, s: NDArray, Vt: NDArray,
) -> NDArray[np.float64]:
    return (U * s) @ Vt


def spectral_norm(A: NDArray[np.float64]) -> float:
    return float(np.linalg.svd(A, compute_uv=False)[0])


def condition_number(A: NDArray[np.float64]) -> float:
    s = np.linalg.svd(A, compute_uv=False)
    return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")


__all__ = [
    "truncated_svd", "randomized_svd", "low_rank_reconstruct",
    "spectral_norm", "condition_number",
]
