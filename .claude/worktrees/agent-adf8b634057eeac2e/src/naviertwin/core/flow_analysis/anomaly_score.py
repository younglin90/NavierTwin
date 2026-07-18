"""다변량 이상치 점수 — Mahalanobis, Local Outlier Factor (단순), Isolation 깊이.

CFD 결과에서 비정상 데이터 포인트(센서 오작동, 시뮬레이션 분기, regime
이탈)를 정량적으로 식별. ROM/AI 입력 검증의 핵심.

상용 툴 대응:
    - SciKit-Learn: LocalOutlierFactor, IsolationForest, EllipticEnvelope
    - 학술: Breunig et al., "LOF: Identifying Density-Based Local Outliers",
      ACM SIGMOD, 2000.

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 3))
    >>> X[0] = [10.0, 10.0, 10.0]  # 명백한 outlier
    >>> from naviertwin.core.flow_analysis.anomaly_score import (
    ...     mahalanobis_score
    ... )
    >>> scores = mahalanobis_score(X)
    >>> int(np.argmax(scores))
    0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def mahalanobis_score(
    X: NDArray[np.float64],
    reference: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Mahalanobis 거리 — 평균 + 공분산 기반 다변량 이상치 점수.

    M(x) = √((x - μ)ᵀ Σ⁻¹ (x - μ)).

    Args:
        X: (N, d) 점.
        reference: (M, d) 기준 분포. None이면 X 자체.

    Returns:
        (N,) 점수.

    Raises:
        ValueError: X가 2D 아님.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    R = X if reference is None else np.asarray(reference, dtype=np.float64)
    if R.ndim != 2 or R.shape[1] != X.shape[1]:
        raise ValueError(
            f"reference shape {R.shape} incompatible with X {X.shape}"
        )

    mu = R.mean(axis=0)
    cov = np.cov(R.T) + 1e-12 * np.eye(R.shape[1])
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov)
    diff = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))


def lof_score(
    X: NDArray[np.float64],
    k: int = 5,
) -> NDArray[np.float64]:
    """Local Outlier Factor — Breunig 2000.

    각 점의 LOF는 이웃 점들의 local density 대비 자신의 density 비율.
    > 1 이면 outlier 가능성.

    Args:
        X: (N, d) 점.
        k: 이웃 수.

    Returns:
        (N,) LOF 점수.

    Raises:
        ValueError: 매개변수 오류.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    N = X.shape[0]
    if k < 1 or k >= N:
        raise ValueError(f"k must be in [1, {N - 1}], got {k}")

    # pairwise 거리
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    # k-th nearest 거리 (k-distance)
    knn_idx = np.argsort(d, axis=1)[:, :k]
    k_dist = np.sort(d, axis=1)[:, k - 1]
    # reachability dist: rd(x, y) = max(k-dist(y), d(x, y))
    rd = np.maximum(k_dist[knn_idx], np.take_along_axis(d, knn_idx, axis=1))
    # local reachability density
    lrd = 1.0 / (rd.mean(axis=1) + 1e-30)
    # LOF
    return lrd[knn_idx].mean(axis=1) / (lrd + 1e-30)


def isolation_depth(
    X: NDArray[np.float64],
    n_trees: int = 50,
    sample_size: int = 256,
    seed: int | None = 0,
) -> NDArray[np.float64]:
    """단순 Isolation Forest 깊이 (Liu 2008 단순화).

    각 점이 무작위 분할 트리에서 격리되는 깊이의 평균.
    얕은 깊이 = outlier.

    Args:
        X: (N, d) 점.
        n_trees: 트리 수.
        sample_size: 트리당 표본 크기.
        seed: RNG.

    Returns:
        (N,) 평균 깊이 (작을수록 outlier).

    Raises:
        ValueError: 매개변수 오류.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    N, d = X.shape
    if n_trees <= 0 or sample_size <= 0:
        raise ValueError(
            f"n_trees and sample_size > 0 required, got {n_trees}/{sample_size}"
        )
    sample_size = min(sample_size, N)
    rng = np.random.default_rng(seed)

    depths = np.zeros((n_trees, N))
    t = 0
    while t < n_trees:
        sample_idx = rng.choice(N, size=sample_size, replace=False)
        X_sub = X[sample_idx]
        # 모든 점에 대해 깊이 측정
        i = 0
        while i < N:
            depths[t, i] = _isolation_depth_single(X[i], X_sub, rng)
            i += 1
        t += 1

    return depths.mean(axis=0)


def _isolation_depth_single(
    x: NDArray[np.float64],
    sample: NDArray[np.float64],
    rng: np.random.Generator,
    max_depth: int = 100,
) -> int:
    """한 점이 sample에서 isolation 되는 깊이."""
    region = sample.copy()
    depth = 0
    while len(region) > 1 and depth < max_depth:
        depth += 1
        # 무작위 차원 + 무작위 분할
        d = region.shape[1]
        feat = int(rng.integers(d))
        lo, hi = float(region[:, feat].min()), float(region[:, feat].max())
        if hi - lo < 1e-30:
            break
        thresh = float(rng.uniform(lo, hi))
        if x[feat] < thresh:
            region = region[region[:, feat] < thresh]
        else:
            region = region[region[:, feat] >= thresh]
    return depth


def z_score_max(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """변수별 z-score의 최대 절댓값 — 단순 outlier 점수.

    Args:
        X: (N, d).

    Returns:
        (N,) max(|z_i|).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-30
    z = (X - mu) / sigma
    return np.max(np.abs(z), axis=1)


def hampel_score_1d(
    x: NDArray[np.float64],
    window: int = 7,
) -> NDArray[np.float64]:
    """1D 이동 윈도우 MAD 점수 — Hampel filter score.

    각 점의 (|x - median(window)|) / (MAD(window) · 1.4826).

    Args:
        x: 1D 시계열.
        window: 윈도우 크기 (홀수).

    Returns:
        (N,) z-score-like 값.

    Raises:
        ValueError: window 오류.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if window < 3 or window % 2 == 0:
        raise ValueError(
            f"window must be odd and >= 3, got {window}"
        )
    N = len(x)
    half = window // 2
    padded = np.pad(x, half, mode="reflect")
    wins = np.lib.stride_tricks.sliding_window_view(padded, window)[:N]
    med = np.median(wins, axis=1)
    mad = np.median(np.abs(wins - med[:, np.newaxis]), axis=1) * 1.4826
    out = np.divide(
        np.abs(x - med),
        mad,
        out=np.zeros(N, dtype=np.float64),
        where=mad >= 1e-30,
    )
    return out


__all__ = [
    "mahalanobis_score",
    "lof_score",
    "isolation_depth",
    "z_score_max",
    "hampel_score_1d",
]
