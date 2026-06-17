"""Active learning — 분산 기반 다음 파라미터 선택.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.active_learning.query import top_variance_query
    >>> vars = np.array([0.1, 0.3, 0.8, 0.2])
    >>> top_variance_query(vars, k=2).tolist()
    [2, 1]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels


def top_variance_query(
    variances: NDArray[np.float64], k: int = 1,
) -> NDArray[np.int64]:
    """분산 높은 상위 k 인덱스."""
    v = np.asarray(variances, dtype=np.float64).ravel()
    k = int(min(k, v.size))
    idx = np.argpartition(-v, k - 1)[:k]
    return idx[np.argsort(-v[idx])]


def greedy_maxmin(
    candidates: NDArray[np.float64], existing: NDArray[np.float64], k: int = 1,
) -> NDArray[np.int64]:
    """이미 샘플된 점에서 가장 먼 후보 k개 (최소 거리 최대화)."""
    c = np.asarray(candidates, dtype=np.float64)
    if existing.size == 0:
        return np.arange(min(k, c.shape[0]))
    e = np.asarray(existing, dtype=np.float64)
    d = np.min(
        np.linalg.norm(c[:, None, :] - e[None, :, :], axis=2), axis=1,
    )
    k = int(min(k, c.shape[0]))
    idx = np.argpartition(-d, k - 1)[:k]
    return idx[np.argsort(-d[idx])]


def expected_improvement(
    mu: NDArray[np.float64], sigma: NDArray[np.float64],
    y_best: float, *, xi: float = 0.0,
) -> NDArray[np.float64]:
    """EI = (μ − y_best − ξ) Φ(z) + σ φ(z). minimization 기준.

    EI > 0 일 때 개선 기대.
    """
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by expected_improvement")
    return _kernels.expected_improvement(mu, sigma, y_best, xi)


__all__ = ["top_variance_query", "greedy_maxmin", "expected_improvement"]
