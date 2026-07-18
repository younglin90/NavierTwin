"""상관관계 분석 — Pearson / Spearman / 공분산 / cross-correlation.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.correlation import pearson_matrix
    >>> X = np.array([[1., 2., 3.], [2., 4., 6.], [1., 0., -1.]])
    >>> C = pearson_matrix(X)
    >>> C.shape
    (3, 3)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pearson_matrix(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """(n_vars, n_samples) → (n_vars, n_vars) Pearson."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_vars, n_samples)")
    return np.corrcoef(X)


def spearman_matrix(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """순위 기반 Spearman."""
    X = np.asarray(X, dtype=np.float64)
    R = np.apply_along_axis(_rank, 1, X)
    return np.corrcoef(R)


def _rank(v: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(np.argsort(v))
    return order.astype(np.float64)


def cross_correlation(
    a: NDArray[np.float64], b: NDArray[np.float64], *, max_lag: int | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """a, b 시계열 간 지연 교차상관 (정규화).

    Returns:
        (lags, corr) — len = 2*max_lag+1.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError("a/b 길이 동일해야 함")
    n = a.size
    a0 = a - a.mean()
    b0 = b - b.mean()
    full = np.correlate(a0, b0, mode="full")
    denom = np.sqrt(np.sum(a0 ** 2) * np.sum(b0 ** 2)) + 1e-30
    corr = full / denom
    lags = np.arange(-n + 1, n)
    if max_lag is not None:
        m = int(max_lag)
        mid = n - 1
        lags = lags[mid - m: mid + m + 1]
        corr = corr[mid - m: mid + m + 1]
    return lags, corr


__all__ = ["pearson_matrix", "spearman_matrix", "cross_correlation"]
