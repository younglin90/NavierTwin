"""Active learning 획득 함수 — EI, PoI, UCB, MES, qBC 등.

Surrogate 모델의 (mu, sigma) 예측에서 다음 평가할 점을 선택. ROM/Kriging
정제, Bayesian Optimization, ROM 재학습 트리거 등에 사용.

상용 툴 대응:
    - SciKit-Optimize (skopt)
    - BoTorch
    - SMT toolbox: EGO with EI

References:
    Mockus, J., "Bayesian Approach to Global Optimization", Kluwer, 1989.
    Srinivas et al., "Gaussian process optimization in the bandit setting",
    ICML, 2010.

Examples:
    >>> import numpy as np
    >>> mu = np.array([1.0, 2.0, 3.0])
    >>> sigma = np.array([0.1, 0.5, 0.2])
    >>> from naviertwin.core.active_learning.acquisition import expected_improvement
    >>> ei = expected_improvement(mu, sigma, y_best=0.5)
    >>> ei.shape
    (3,)
"""

from __future__ import annotations

from math import erf, sqrt

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _norm_pdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)


def _norm_cdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([0.5 * (1.0 + erf(xi / sqrt(2.0))) for xi in np.atleast_1d(x)])


def expected_improvement(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    y_best: float,
    xi: float = 0.0,
    minimize: bool = True,
) -> NDArray[np.float64]:
    """Expected Improvement EI(x) = E[max(y_best - μ - ξ, 0)].

    Args:
        mu: (N,) 예측 평균.
        sigma: (N,) 예측 표준편차.
        y_best: 현재 최적값.
        xi: 탐색 가중 (큰 ξ = 더 탐색).
        minimize: True면 최소화, False면 최대화.

    Returns:
        (N,) EI 배열. 클수록 좋은 후보.

    Raises:
        ValueError: 형상 불일치 또는 음수 sigma.
    """
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    if mu.shape != sigma.shape:
        raise ValueError(f"shape mismatch: {mu.shape} vs {sigma.shape}")
    if np.any(sigma < 0):
        raise ValueError("sigma must be non-negative")

    if minimize:
        improvement = y_best - mu - xi
    else:
        improvement = mu - y_best - xi

    safe_sigma = np.maximum(sigma, 1e-30)
    z = improvement / safe_sigma
    Phi = _norm_cdf(z)
    phi = _norm_pdf(z)
    ei = improvement * Phi + sigma * phi
    # sigma == 0인 점은 EI = max(improvement, 0)
    ei = np.where(sigma > 1e-30, ei, np.maximum(improvement, 0.0))
    return ei


def probability_of_improvement(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    y_best: float,
    xi: float = 0.01,
    minimize: bool = True,
) -> NDArray[np.float64]:
    """PoI(x) = P[y < y_best - ξ] (minimization).

    Args:
        mu, sigma: 예측 (μ, σ).
        y_best: 현재 최적.
        xi: 임계 마진.
        minimize: True/False.

    Returns:
        (N,) ∈ [0, 1].
    """
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    if mu.shape != sigma.shape:
        raise ValueError(f"shape mismatch: {mu.shape} vs {sigma.shape}")
    if np.any(sigma < 0):
        raise ValueError("sigma must be non-negative")

    if minimize:
        z = (y_best - mu - xi) / np.maximum(sigma, 1e-30)
    else:
        z = (mu - y_best - xi) / np.maximum(sigma, 1e-30)
    return _norm_cdf(z)


def upper_confidence_bound(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    kappa: float = 2.0,
) -> NDArray[np.float64]:
    """UCB = μ + κ·σ. 최대화 대상.

    Args:
        mu, sigma: 예측.
        kappa: 탐색 파라미터 (큰 κ → 더 탐색).

    Returns:
        (N,) UCB.

    Raises:
        ValueError: 형상 불일치 또는 sigma 음수.
    """
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    if mu.shape != sigma.shape:
        raise ValueError(f"shape mismatch: {mu.shape} vs {sigma.shape}")
    if np.any(sigma < 0):
        raise ValueError("sigma must be non-negative")
    if kappa < 0:
        raise ValueError(f"kappa must be >= 0, got {kappa}")
    return mu + kappa * sigma


def lower_confidence_bound(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    kappa: float = 2.0,
) -> NDArray[np.float64]:
    """LCB = μ - κ·σ. 최소화 대상."""
    return mu - kappa * np.asarray(sigma, dtype=np.float64).ravel()


def thompson_sample(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Thompson sampling: ŷ ~ N(μ, σ²).

    각 후보의 추출된 표본을 이용해 best를 선택 (random tie-breaking).

    Args:
        mu, sigma: 예측.
        seed: RNG seed.

    Returns:
        (N,) 표본 배열.
    """
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    if mu.shape != sigma.shape:
        raise ValueError(f"shape mismatch: {mu.shape} vs {sigma.shape}")
    rng = np.random.default_rng(seed)
    return mu + sigma * rng.standard_normal(mu.shape)


def max_variance_query(
    sigma: NDArray[np.float64],
    n_select: int = 1,
) -> NDArray[np.intp]:
    """예측 분산 최대화 — 모델 불확실성 감소 (탐색).

    Args:
        sigma: (N,) 예측 표준편차.
        n_select: 선택할 개수.

    Returns:
        (n_select,) 인덱스 (분산 큰 순).
    """
    s = np.asarray(sigma, dtype=np.float64).ravel()
    if n_select <= 0 or n_select > s.size:
        raise ValueError(f"n_select out of range, got {n_select}")
    idx = np.argsort(-s)
    return idx[:n_select].astype(np.intp)


def query_by_committee(
    predictions: NDArray[np.float64],
    n_select: int = 1,
) -> NDArray[np.intp]:
    """위원회 분산 (Query-by-Committee) — 위원 예측의 분산이 큰 점 선택.

    Args:
        predictions: (M, N) 위원별 예측 (M 위원, N 후보).
        n_select: 선택할 후보 수.

    Returns:
        (n_select,) 인덱스.

    Raises:
        ValueError: predictions가 2D 아님.
    """
    P = np.asarray(predictions, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError(f"predictions must be 2D, got {P.shape}")
    var = P.var(axis=0)
    if n_select <= 0 or n_select > P.shape[1]:
        raise ValueError(f"n_select out of range, got {n_select}")
    idx = np.argsort(-var)
    return idx[:n_select].astype(np.intp)


def greedy_batch_acquisition(
    acq: NDArray[np.float64],
    candidates: NDArray[np.float64],
    batch_size: int,
    min_distance: float | None = None,
) -> NDArray[np.intp]:
    """배치 acquisition — 가장 큰 acq 점들을 거리 제약 하에 선택.

    Args:
        acq: (N,) acquisition 값.
        candidates: (N, D) 후보 좌표.
        batch_size: 선택할 점 수.
        min_distance: 최소 거리 제약 (None이면 거리 무시).

    Returns:
        (batch_size,) 정수 인덱스.

    Raises:
        ValueError: 매개변수 오류.
    """
    a = np.asarray(acq, dtype=np.float64).ravel()
    C = np.asarray(candidates, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != a.shape[0]:
        raise ValueError(
            f"candidates shape mismatch: {C.shape} vs {a.shape}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size > 0 required, got {batch_size}")

    sorted_idx = np.argsort(-a)
    selected: list[int] = []
    for idx in sorted_idx:
        if len(selected) >= batch_size:
            break
        if min_distance is not None and selected:
            dists = np.linalg.norm(C[idx] - C[selected], axis=1)
            if dists.min() < min_distance:
                continue
        selected.append(int(idx))
    return np.array(selected, dtype=np.intp)


__all__ = [
    "expected_improvement",
    "probability_of_improvement",
    "upper_confidence_bound",
    "lower_confidence_bound",
    "thompson_sample",
    "max_variance_query",
    "query_by_committee",
    "greedy_batch_acquisition",
]
