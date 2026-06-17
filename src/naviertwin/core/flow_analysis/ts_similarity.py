"""시계열 유사성 검색 — Shape-Based Distance, MASS, sliding-window matching.

CFD 시계열 (프로브, 모드 계수)에서 패턴 유사한 구간을 빠르게 찾는다.
모티프 검색, 이상 패턴 매칭, 클러스터링 거리 함수에 사용.

상용 툴 대응:
    - tslearn: shape_based_distance, MatrixProfile
    - stumpy: stump (Matrix Profile)
    - 학술: Yeh et al., "Matrix Profile I", ICDM 2016.
            Mueen et al., "MASS: Mueen's Similarity Search Algorithm".

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> series = rng.standard_normal(1000)
    >>> query = series[100:130].copy()
    >>> from naviertwin.core.flow_analysis.ts_similarity import mass_search
    >>> dist = mass_search(query, series)
    >>> int(np.argmin(dist))  # 100 근처
    100
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def shape_based_distance(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float:
    """Shape-Based Distance (SBD) — z-normalize 후 정규화 cross-correlation 최대값 기반.

    SBD(x, y) = 1 - max(NCC(x, y)).
    값 범위: [0, 2], 0 = 완전 일치.

    Args:
        x, y: 1D 시계열 (길이 동일).

    Returns:
        SBD ∈ [0, 2].

    Raises:
        ValueError: 길이 불일치 또는 너무 짧음.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if len(x) < 2:
        raise ValueError(f"need at least 2 points, got {len(x)}")

    # z-normalize
    sx = x.std()
    sy = y.std()
    if sx < 1e-30 or sy < 1e-30:
        return float(np.linalg.norm(x - y) / max(len(x), 1))
    xn = (x - x.mean()) / sx
    yn = (y - y.mean()) / sy

    # 모든 lag에 대한 정규화 상관
    n = len(xn)
    cc = np.correlate(xn, yn, mode="full") / n
    return float(1.0 - np.max(cc))


def mass_search(
    query: NDArray[np.float64],
    series: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Mueen's Similarity Search Algorithm (MASS) — 정규화 거리 프로파일.

    각 위치 i에서 series[i:i+len(query)]와 query의 z-정규화 유클리드 거리.
    O(n log n) FFT 기반.

    Args:
        query: (m,) 패턴.
        series: (n,) 검색 대상 (n ≥ m).

    Returns:
        (n - m + 1,) 거리 프로파일.

    Raises:
        ValueError: 입력 형상 오류.
    """
    q = np.asarray(query, dtype=np.float64).ravel()
    s = np.asarray(series, dtype=np.float64).ravel()
    m = len(q)
    n = len(s)
    if m < 2:
        raise ValueError(f"query length must be >= 2, got {m}")
    if n < m:
        raise ValueError(f"series length {n} < query length {m}")

    # query z-normalize
    mu_q = q.mean()
    sigma_q = q.std()
    if sigma_q < 1e-30:
        sigma_q = 1.0
    q_norm = (q - mu_q) / sigma_q

    # series에서 sliding mean / std
    cumsum = np.concatenate([[0.0], np.cumsum(s)])
    cumsum_sq = np.concatenate([[0.0], np.cumsum(s * s)])
    seg_sum = cumsum[m:] - cumsum[:-m]
    seg_sumsq = cumsum_sq[m:] - cumsum_sq[:-m]
    mu_s = seg_sum / m
    var_s = seg_sumsq / m - mu_s ** 2
    sigma_s = np.sqrt(np.maximum(var_s, 0.0))
    sigma_s_safe = np.where(sigma_s < 1e-30, 1.0, sigma_s)

    # cross-correlation via FFT
    # corr[i] = sum_{k=0}^{m-1} s[i+k] * q[k]
    qpad = np.zeros(n)
    qpad[:m] = q_norm[::-1]
    Q_fft = np.fft.fft(qpad)
    S_fft = np.fft.fft(s)
    corr_full = np.real(np.fft.ifft(S_fft * Q_fft))
    corr = corr_full[m - 1 : n]  # length n-m+1

    # 정규화 거리: D² = 2m (1 - (corr - m·mu_s·mu_q_norm)/(m·sigma_s · sigma_q_norm))
    # q_norm은 이미 z-normalized라 mean≈0, std≈1
    dist_sq = 2.0 * m * (1.0 - corr / (m * sigma_s_safe))
    dist_sq = np.maximum(dist_sq, 0.0)
    return np.sqrt(dist_sq)


def find_top_k_motifs(
    series: NDArray[np.float64],
    window: int,
    k: int = 1,
    exclusion_radius: int | None = None,
) -> list[tuple[int, int, float]]:
    """Top-k 모티프 쌍 검색 — Matrix Profile 단순 변종.

    각 위치를 query로 한 거리 프로파일에서 가장 가까운 (자기 제외)
    위치 쌍을 모아 거리가 작은 순으로 k개.

    Args:
        series: (N,) 시계열.
        window: 모티프 길이.
        k: 반환할 쌍 수.
        exclusion_radius: 자기-매칭 제외 반경. None이면 window/2.

    Returns:
        list of (idx_a, idx_b, distance) 거리 오름차순.

    Raises:
        ValueError: 매개변수 오류.
    """
    s = np.asarray(series, dtype=np.float64).ravel()
    N = len(s)
    if window < 2 or window > N // 2:
        raise ValueError(f"window in [2, {N // 2}], got {window}")
    if k <= 0:
        raise ValueError(f"k > 0, got {k}")
    if exclusion_radius is None:
        exclusion_radius = window // 2

    n_pos = N - window + 1
    best_dist = np.full(n_pos, np.inf)
    best_idx = np.full(n_pos, -1, dtype=int)

    i = 0
    while i < n_pos:
        query = s[i : i + window]
        dist = mass_search(query, s)
        # 자기 + 인접 제외
        lo = max(0, i - exclusion_radius)
        hi = min(n_pos, i + exclusion_radius + 1)
        dist[lo:hi] = np.inf
        j = int(np.argmin(dist))
        if dist[j] < best_dist[i]:
            best_dist[i] = dist[j]
            best_idx[i] = j
        i += 1

    # 가장 작은 거리 k개 (대칭 쌍 중복 제거)
    pairs: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    order = np.argsort(best_dist)
    order_pos = 0
    while order_pos < order.size:
        i = order[order_pos]
        if best_idx[i] < 0:
            order_pos += 1
            continue
        a, b = (int(i), int(best_idx[i]))
        pair = (min(a, b), max(a, b))
        if pair in seen:
            order_pos += 1
            continue
        seen.add(pair)
        pairs.append((a, b, float(best_dist[i])))
        if len(pairs) >= k:
            break
        order_pos += 1
    return pairs


def template_matching(
    template: NDArray[np.float64],
    series: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.intp]:
    """임계값 이하의 모든 매칭 위치.

    Args:
        template: 패턴.
        series: 검색 대상.
        threshold: 매칭 거리 임계값.

    Returns:
        매칭된 시작 인덱스 배열.

    Raises:
        ValueError: threshold ≤ 0.
    """
    if threshold <= 0:
        raise ValueError(f"threshold > 0, got {threshold}")
    dist = mass_search(template, series)
    return np.where(dist <= threshold)[0].astype(np.intp)


__all__ = [
    "shape_based_distance",
    "mass_search",
    "find_top_k_motifs",
    "template_matching",
]
