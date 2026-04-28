"""POD 계수 시계열 (trajectory) 클러스터링 — 동작 모드/regime 식별.

CFD 결과의 시간 진화를 POD 계수 공간 trajectory로 보고, 비슷한 동역학
(stall, surge, normal flow 등)을 가진 시간 구간을 군집화. ROM/AI에서
regime-switching 모델 또는 fault detection에 사용.

상용 툴 대응:
    - MATLAB: kmeans + dtw
    - tslearn: TimeSeriesKMeans
    - 학술: Aghabozorgi et al., "Time-series clustering — A decade review",
      Information Systems, 2015.

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> coeffs = rng.standard_normal((100, 5))  # (n_t, n_modes)
    >>> from naviertwin.core.dimensionality_reduction.trajectory_clustering import (
    ...     window_kmeans
    ... )
    >>> labels, centers = window_kmeans(coeffs, window=20, n_clusters=3)
    >>> labels.shape[0] == 100 - 20 + 1
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def window_kmeans(
    coeffs: NDArray[np.float64],
    window: int,
    n_clusters: int,
    max_iter: int = 100,
    seed: int | None = 0,
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """슬라이딩 윈도우 K-means: 각 윈도우의 평균 계수를 클러스터링.

    Args:
        coeffs: (n_t, n_modes) POD 계수 시계열.
        window: 윈도우 크기.
        n_clusters: 클러스터 수.
        max_iter: K-means 최대 반복.
        seed: RNG seed.

    Returns:
        (labels, centers): labels (n_t - window + 1,), centers (n_clusters, n_modes).

    Raises:
        ValueError: 매개변수 오류.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.ndim != 2:
        raise ValueError(f"coeffs must be 2D, got {coeffs.shape}")
    n_t, n_modes = coeffs.shape
    if window <= 0 or window > n_t:
        raise ValueError(f"window in [1, {n_t}], got {window}")
    if n_clusters <= 0:
        raise ValueError(f"n_clusters > 0, got {n_clusters}")

    n_win = n_t - window + 1
    features = np.zeros((n_win, n_modes))
    for i in range(n_win):
        features[i] = coeffs[i : i + window].mean(axis=0)

    return _kmeans(features, n_clusters, max_iter=max_iter, seed=seed)


def _kmeans(
    X: NDArray[np.float64],
    k: int,
    max_iter: int = 100,
    seed: int | None = 0,
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """단순 K-means (Lloyd 알고리즘) — k++ 초기화."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    if k > n:
        raise ValueError(f"n_clusters {k} > n_samples {n}")

    # k-means++
    centers = np.zeros((k, d))
    centers[0] = X[rng.integers(n)]
    for c in range(1, k):
        d_min = np.min(np.linalg.norm(X[:, None, :] - centers[None, :c, :], axis=2), axis=1)
        prob = d_min ** 2
        prob /= prob.sum() + 1e-30
        centers[c] = X[rng.choice(n, p=prob)]

    labels = np.zeros(n, dtype=np.intp)
    for _ in range(max_iter):
        # assignment
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.intp)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # update
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = X[mask].mean(axis=0)

    return labels, centers


def trajectory_distance_matrix(
    trajectories: list[NDArray[np.float64]],
    metric: str = "euclidean_avg",
) -> NDArray[np.float64]:
    """N개 trajectory 사이의 pairwise 거리 행렬.

    Args:
        trajectories: 길이 N 리스트, 각 (n_t, n_modes).
        metric: "euclidean_avg" (시간평균 후 거리), "endpoint" (시작-끝점 거리),
                "frobenius" (전체 차이의 Frobenius 노름; 같은 길이 필요).

    Returns:
        (N, N) 대칭 거리 행렬.

    Raises:
        ValueError: metric 오류 또는 길이 불일치.
    """
    N = len(trajectories)
    if N == 0:
        return np.zeros((0, 0))
    if metric not in ("euclidean_avg", "endpoint", "frobenius"):
        raise ValueError(f"metric '{metric}' invalid")

    D = np.zeros((N, N))
    if metric == "euclidean_avg":
        means = np.array([t.mean(axis=0) for t in trajectories])
        for i in range(N):
            for j in range(i + 1, N):
                D[i, j] = float(np.linalg.norm(means[i] - means[j]))
                D[j, i] = D[i, j]
    elif metric == "endpoint":
        ends = np.array([t[-1] for t in trajectories])
        for i in range(N):
            for j in range(i + 1, N):
                D[i, j] = float(np.linalg.norm(ends[i] - ends[j]))
                D[j, i] = D[i, j]
    else:  # frobenius
        # 모든 길이가 같다고 가정
        L = trajectories[0].shape
        for t in trajectories:
            if t.shape != L:
                raise ValueError(
                    f"trajectories must all have same shape for frobenius; got {t.shape} vs {L}"
                )
        for i in range(N):
            for j in range(i + 1, N):
                D[i, j] = float(np.linalg.norm(trajectories[i] - trajectories[j]))
                D[j, i] = D[i, j]
    return D


def cluster_silhouette(
    X: NDArray[np.float64],
    labels: NDArray[np.intp],
) -> float:
    """평균 실루엣 계수 (cluster quality).

    s_i = (b_i - a_i) / max(a_i, b_i), where:
        a_i = 같은 클러스터 내 평균 거리.
        b_i = 가장 가까운 다른 클러스터 평균 거리.

    Args:
        X: (N, d) 데이터.
        labels: (N,) 클러스터 라벨.

    Returns:
        평균 실루엣 ∈ [-1, 1] (1 가까울수록 좋음).

    Raises:
        ValueError: 형상 불일치 또는 클러스터 < 2.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.intp)
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels length mismatch")
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0

    n = X.shape[0]
    s_arr = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if same.sum() == 0:
            continue
        a_i = float(np.mean(np.linalg.norm(X[same] - X[i], axis=1)))
        # b_i: 다른 클러스터 중 평균 거리 최소
        b_vals = []
        for c in unique:
            if c == labels[i]:
                continue
            other = labels == c
            if other.sum() == 0:
                continue
            b_vals.append(np.mean(np.linalg.norm(X[other] - X[i], axis=1)))
        if not b_vals:
            continue
        b_i = float(min(b_vals))
        s_arr[i] = (b_i - a_i) / max(a_i, b_i, 1e-30)

    return float(s_arr.mean())


def label_runs(
    labels: NDArray[np.intp],
) -> list[tuple[int, int, int]]:
    """라벨의 연속 구간 추출.

    Args:
        labels: 시간 순 라벨 시퀀스.

    Returns:
        list of (label, start, end) — half-open [start, end).
    """
    labels = np.asarray(labels, dtype=np.intp).ravel()
    if len(labels) == 0:
        return []
    runs: list[tuple[int, int, int]] = []
    current = int(labels[0])
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != current:
            runs.append((current, start, i))
            current = int(labels[i])
            start = i
    runs.append((current, start, len(labels)))
    return runs


__all__ = [
    "window_kmeans",
    "trajectory_distance_matrix",
    "cluster_silhouette",
    "label_runs",
]
