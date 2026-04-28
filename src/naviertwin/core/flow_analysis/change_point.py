"""변화점 검출 (Change-Point Detection) — PELT, Binary Segmentation, Window.

CFD 시계열에서 통계적 특성이 갑작스럽게 변하는 시점을 자동 식별.
flow regime 전환, fault onset, simulation convergence 시작점 검출.

상용 툴 대응:
    - R `changepoint` 패키지: PELT, BinSeg, SegNeigh
    - Python `ruptures`: Pelt, Binseg, Window
    - 학술: Killick et al., "Optimal Detection of Changepoints With a
      Linear Computational Cost", JASA 2012.

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> # 분명한 변화: 처음 50개 N(0,1), 다음 50개 N(5,1)
    >>> x = np.concatenate([rng.standard_normal(50), 5 + rng.standard_normal(50)])
    >>> from naviertwin.core.flow_analysis.change_point import binary_segmentation
    >>> cps = binary_segmentation(x, n_changepoints=1)
    >>> abs(cps[0] - 50) < 5
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _segment_cost(x: NDArray[np.float64], start: int, end: int) -> float:
    """L2 cost C(start:end) = Σ (x_i - mean)² (분산 기반)."""
    if end - start < 2:
        return 0.0
    seg = x[start:end]
    return float(np.sum((seg - seg.mean()) ** 2))


def binary_segmentation(
    x: NDArray[np.float64],
    n_changepoints: int = 1,
    min_size: int = 5,
) -> list[int]:
    """이진 분할 — 가장 큰 cost 감소 위치를 재귀 분할.

    O(n²) 단순 구현. 큰 데이터에는 PELT 권장.

    Args:
        x: (N,) 1D 시계열.
        n_changepoints: 검출할 변화점 수.
        min_size: 최소 세그먼트 크기.

    Returns:
        오름차순 변화점 인덱스 리스트.

    Raises:
        ValueError: 매개변수 오류.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    N = len(x)
    if N < 2 * min_size:
        raise ValueError(
            f"signal too short: {N} < 2 * min_size ({2 * min_size})"
        )
    if n_changepoints <= 0:
        raise ValueError(f"n_changepoints > 0, got {n_changepoints}")
    if min_size < 1:
        raise ValueError(f"min_size >= 1, got {min_size}")

    cps: list[int] = []
    segments = [(0, N)]
    for _ in range(n_changepoints):
        best_gain = -np.inf
        best_cp = -1
        best_seg_idx = -1
        for s_idx, (start, end) in enumerate(segments):
            base = _segment_cost(x, start, end)
            for k in range(start + min_size, end - min_size + 1):
                gain = base - _segment_cost(x, start, k) - _segment_cost(x, k, end)
                if gain > best_gain:
                    best_gain = gain
                    best_cp = k
                    best_seg_idx = s_idx
        if best_cp < 0 or best_gain <= 0:
            break
        cps.append(best_cp)
        s, e = segments.pop(best_seg_idx)
        segments.append((s, best_cp))
        segments.append((best_cp, e))

    return sorted(cps)


def pelt(
    x: NDArray[np.float64],
    penalty: float | None = None,
    min_size: int = 5,
) -> list[int]:
    """PELT (Pruned Exact Linear Time) 알고리즘 — Killick et al. 2012.

    Best segmentation with penalty β: F(n) = min over τ [F(τ) + C(τ:n) + β].

    Args:
        x: 1D 시계열.
        penalty: 변화점 패널티 β. None이면 BIC: β = 2σ² log N.
        min_size: 최소 세그먼트 크기.

    Returns:
        변화점 인덱스 리스트 (오름차순).

    Raises:
        ValueError: 매개변수 오류.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    N = len(x)
    if N < 2 * min_size:
        raise ValueError(f"signal too short: {N}")
    if min_size < 1:
        raise ValueError(f"min_size >= 1, got {min_size}")

    if penalty is None:
        sigma2 = float(np.var(x) + 1e-30)
        penalty = 2.0 * sigma2 * np.log(N)

    F = np.full(N + 1, np.inf)
    F[0] = -penalty
    last = np.zeros(N + 1, dtype=int)
    candidates = [0]

    for tau in range(min_size, N + 1):
        best = np.inf
        best_t = 0
        new_candidates = []
        for t in candidates:
            if tau - t < min_size:
                new_candidates.append(t)
                continue
            cost = F[t] + _segment_cost(x, t, tau) + penalty
            if cost < best:
                best = cost
                best_t = t
            # pruning condition (Killick): keep t if F[t] + cost(t,tau) ≤ F[tau]
            if F[t] + _segment_cost(x, t, tau) <= F[tau] + penalty + 1e-12:
                new_candidates.append(t)
        F[tau] = best
        last[tau] = best_t
        if tau not in new_candidates:
            new_candidates.append(tau)
        candidates = new_candidates

    # 역추적
    cps: list[int] = []
    cur = last[N]
    while cur > 0:
        cps.append(int(cur))
        cur = last[cur]
    return sorted(cps)


def window_method(
    x: NDArray[np.float64],
    width: int = 20,
    threshold: float = 3.0,
) -> list[int]:
    """슬라이딩 윈도우 평균 차이 검출 — 단순/빠른 휴리스틱.

    각 위치에서 좌우 width 평균의 차이 ≥ threshold·σ 이면 변화점.

    Args:
        x: 1D 시계열.
        width: 좌우 윈도우 크기.
        threshold: σ 단위 임계값.

    Returns:
        변화점 인덱스 리스트.

    Raises:
        ValueError: width ≤ 0 또는 threshold ≤ 0.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if width <= 0:
        raise ValueError(f"width > 0, got {width}")
    if threshold <= 0:
        raise ValueError(f"threshold > 0, got {threshold}")

    N = len(x)
    sigma = float(x.std()) + 1e-30
    cps: list[int] = []
    for i in range(width, N - width):
        left_mean = x[i - width : i].mean()
        right_mean = x[i : i + width].mean()
        if abs(right_mean - left_mean) > threshold * sigma:
            # NMS: 인접 변화점 제외
            if not cps or i - cps[-1] >= width:
                cps.append(i)
    return cps


def segment_means(
    x: NDArray[np.float64],
    changepoints: list[int],
) -> list[float]:
    """변화점 사이의 각 세그먼트 평균 반환.

    Args:
        x: 1D 시계열.
        changepoints: 정렬된 변화점 인덱스 리스트.

    Returns:
        len(changepoints)+1개 평균 리스트.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    N = len(x)
    if N == 0:
        return []
    boundaries = [0, *sorted(changepoints), N]
    means = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e <= s:
            means.append(0.0)
        else:
            means.append(float(x[s:e].mean()))
    return means


def detection_score(
    x: NDArray[np.float64],
    changepoints: list[int],
) -> float:
    """변화점 검출 품질 점수 — 분산 감소율.

    Score = (var_total - sum(var_segment)) / var_total ∈ [0, 1].

    Args:
        x: 시계열.
        changepoints: 변화점 위치.

    Returns:
        분산 감소율.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    N = len(x)
    if N < 2:
        return 0.0
    var_total = float(np.var(x)) * N
    boundaries = [0, *sorted(changepoints), N]
    var_segments = 0.0
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s > 0:
            seg = x[s:e]
            var_segments += float(np.var(seg)) * (e - s)
    if var_total < 1e-30:
        return 0.0
    return 1.0 - var_segments / var_total


__all__ = [
    "binary_segmentation",
    "pelt",
    "window_method",
    "segment_means",
    "detection_score",
]
