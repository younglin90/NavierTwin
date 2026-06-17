"""통계 수렴 진단 — 시계열 평균/분산이 안정화되었는지 확인.

긴 LES/DNS 시뮬레이션에서 통계가 충분히 수렴했는지 자동 판정.
배치 평균법, Geweke 진단, plateau detection 제공.

상용 툴 대응:
    - Ansys Fluent: Statistics → Sample size convergence
    - Tecplot 360: Time-Average convergence plots
    - 학술 베이스: Welford 1962, Geweke 1992

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> u = 1.0 + rng.standard_normal(10000)
    >>> from naviertwin.core.flow_analysis.stat_convergence import (
    ...     batch_means_se, geweke_diagnostic
    ... )
    >>> mean, se = batch_means_se(u, n_batches=20)
    >>> abs(mean - 1.0) < 0.1
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def batch_means_se(
    x: NDArray[np.float64],
    n_batches: int = 20,
) -> tuple[float, float]:
    """배치 평균 표준오차 추정 — 자기상관 보정 평균/표준오차.

    데이터를 n_batches로 나눠 각 배치 평균으로 분산 추정.
    AR(1) 같은 상관 데이터에서 단순 σ/√N보다 정확.

    Args:
        x: (N,) 시계열.
        n_batches: 배치 수.

    Returns:
        (mean, se): 평균과 표준오차.

    Raises:
        ValueError: n_batches ≤ 1 또는 x 길이 부족.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if n_batches <= 1:
        raise ValueError(f"n_batches must be > 1, got {n_batches}")
    if len(x) < 2 * n_batches:
        raise ValueError(
            f"x length {len(x)} too short with {n_batches} batches"
        )

    batch_size = len(x) // n_batches
    used = batch_size * n_batches
    batches = x[:used].reshape(n_batches, batch_size)
    batch_means = batches.mean(axis=1)

    grand_mean = float(batch_means.mean())
    se = float(batch_means.std(ddof=1) / np.sqrt(n_batches))
    return grand_mean, se


def geweke_diagnostic(
    x: NDArray[np.float64],
    first_frac: float = 0.1,
    last_frac: float = 0.5,
) -> float:
    """Geweke z-score 진단 — 시계열 첫/끝 부분 평균 차이의 표준화.

    |z| < 2 면 수렴, |z| > 2면 비수렴.

    Args:
        x: 시계열.
        first_frac: 첫 부분 비율 (기본 0.1).
        last_frac: 끝 부분 비율 (기본 0.5).

    Returns:
        z-score.

    Raises:
        ValueError: frac 범위 오류.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    N = len(x)
    if not (0 < first_frac < 1) or not (0 < last_frac < 1):
        raise ValueError(
            f"first/last_frac must be in (0, 1), got {first_frac}, {last_frac}"
        )
    if first_frac + last_frac > 1:
        raise ValueError("first_frac + last_frac must be <= 1")

    n_first = max(2, int(first_frac * N))
    n_last = max(2, int(last_frac * N))

    a = x[:n_first]
    b = x[-n_last:]
    var_a = a.var(ddof=1) / n_first + 1e-30
    var_b = b.var(ddof=1) / n_last + 1e-30
    z = (a.mean() - b.mean()) / np.sqrt(var_a + var_b)
    return float(z)


def effective_sample_size(
    x: NDArray[np.float64],
    max_lag: int | None = None,
) -> float:
    """유효 표본 크기 N_eff = N / (1 + 2 Σ ρ_k)— 자기상관 보정.

    Args:
        x: 시계열.
        max_lag: 자기상관 합산 최대 랙. None이면 N // 4.

    Returns:
        N_eff.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by effective_sample_size")
    return float(_kernels.effective_sample_size(x, -1 if max_lag is None else max_lag))


def plateau_detector(
    x: NDArray[np.float64],
    window: int = 100,
    tol_rel: float = 0.01,
) -> int | None:
    """누적 평균이 plateau에 도달한 첫 인덱스 추정.

    cumulative_mean[i+window] - cumulative_mean[i] 변화가
    tol_rel · |cumulative_mean[i+window]| 미만이면 plateau로 간주.

    Args:
        x: 시계열.
        window: 비교 창 크기.
        tol_rel: 상대 허용 오차.

    Returns:
        plateau 시작 인덱스 또는 None (미수렴).

    Raises:
        ValueError: window ≤ 0.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    if len(x) < 2 * window:
        return None

    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by plateau_detector")
    return _kernels.plateau_detector(x, window, tol_rel)


def autocorrelation_time(
    x: NDArray[np.float64],
    max_lag: int | None = None,
) -> float:
    """적분 자기상관 시간 τ_int = 1 + 2 Σ ρ_k.

    Args:
        x: 시계열.
        max_lag: 최대 랙.

    Returns:
        τ_int (단위: 샘플 간격).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by autocorrelation_time")
    return float(_kernels.autocorrelation_time(x, -1 if max_lag is None else max_lag))


__all__ = [
    "batch_means_se",
    "geweke_diagnostic",
    "effective_sample_size",
    "plateau_detector",
    "autocorrelation_time",
]
