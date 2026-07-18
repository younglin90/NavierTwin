"""시계열 특성 추출 — 정량 시간 영역 특성 라이브러리.

CFD/물리 시계열을 압축된 특성 벡터로 표현. 이상 검출, 클러스터링,
ROM 입력 등에 사용. tsfresh/catch22의 경량 대안.

상용 툴 대응:
    - tsfresh: 800+ 특성
    - catch22: 22 robust features
    - MATLAB hctsa: 7000+ 특성

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = np.sin(np.linspace(0, 4 * np.pi, 200)) + 0.1 * rng.standard_normal(200)
    >>> from naviertwin.core.flow_analysis.ts_features import extract_features
    >>> feats = extract_features(x)
    >>> "mean" in feats and "std" in feats
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required by accelerated time-series features")


def _safe_float(value: float, default: float = 0.0) -> float:
    """NaN/Inf safe conversion."""
    if np.isnan(value) or np.isinf(value):
        return default
    return float(value)


def mean_above_quantile(x: NDArray[np.float64], q: float = 0.75) -> float:
    """상위 q 분위 이상의 평균 (heavy-tail 검출)."""
    threshold = np.percentile(x, q * 100)
    above = x[x >= threshold]
    if len(above) == 0:
        return 0.0
    return _safe_float(above.mean())


def long_run_above_mean(x: NDArray[np.float64]) -> int:
    """평균 이상 연속 구간의 최대 길이."""
    above = x > x.mean()
    if not above.any():
        return 0
    diff = np.diff(np.concatenate([[False], above, [False]]).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0:
        return 0
    return int(np.max(ends - starts))


def number_peaks(x: NDArray[np.float64], support: int = 3) -> int:
    """양 이웃이 모두 더 작은 점의 수 (peak count).

    Args:
        x: 시계열.
        support: 좌우 이웃 길이.
    """
    return int(_kernels.number_peaks(np.asarray(x, dtype=np.float64).ravel(), int(support)))


def first_index_above_mean(x: NDArray[np.float64]) -> int:
    """평균을 처음 넘는 위치."""
    above = x > x.mean()
    idx = np.argmax(above)
    if not above.any():
        return -1
    return int(idx)


def percentage_below_zero(x: NDArray[np.float64]) -> float:
    """음수 비율."""
    if len(x) == 0:
        return 0.0
    return float(np.sum(x < 0) / len(x))


def absolute_sum_of_changes(x: NDArray[np.float64]) -> float:
    """Σ |x[t+1] - x[t]| — total variation."""
    if len(x) < 2:
        return 0.0
    return _safe_float(np.sum(np.abs(np.diff(x))))


def mean_absolute_change(x: NDArray[np.float64]) -> float:
    """평균 |Δx|."""
    if len(x) < 2:
        return 0.0
    return _safe_float(np.mean(np.abs(np.diff(x))))


def autocorrelation_lag1(x: NDArray[np.float64]) -> float:
    """ρ(1) — 자기상관 lag-1."""
    if len(x) < 2:
        return 0.0
    xp = x - x.mean()
    var = float(np.dot(xp, xp))
    if var < 1e-30:
        return 0.0
    return _safe_float(np.dot(xp[:-1], xp[1:]) / var)


def trend_slope(x: NDArray[np.float64]) -> float:
    """선형 회귀 기울기 (시간축 = 0..N-1)."""
    if len(x) < 2:
        return 0.0
    t = np.arange(len(x), dtype=np.float64)
    t_mean = t.mean()
    x_mean = x.mean()
    num = np.sum((t - t_mean) * (x - x_mean))
    den = np.sum((t - t_mean) ** 2)
    if den < 1e-30:
        return 0.0
    return _safe_float(num / den)


def crest_factor(x: NDArray[np.float64]) -> float:
    """피크/RMS 비 = max(|x|) / RMS — 충격 신호 식별."""
    rms = np.sqrt(np.mean(x ** 2))
    if rms < 1e-30:
        return 0.0
    return _safe_float(np.max(np.abs(x)) / rms)


def shannon_entropy(x: NDArray[np.float64], bins: int = 10) -> float:
    """히스토그램 기반 Shannon 엔트로피 H = -Σ p log p."""
    if len(x) < 2:
        return 0.0
    counts, _ = np.histogram(x, bins=bins)
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return _safe_float(-np.sum(p * np.log(p)))


def zero_crossing_rate(x: NDArray[np.float64]) -> float:
    """평균 대비 부호 변화 횟수 / 길이."""
    xp = x - x.mean()
    if len(xp) < 2:
        return 0.0
    crosses = np.sum(np.diff(np.sign(xp)) != 0)
    return _safe_float(crosses / (len(xp) - 1))


def extract_features(
    x: NDArray[np.float64],
) -> dict[str, float]:
    """모든 시계열 특성을 dict로 반환.

    Args:
        x: 1D 시계열.

    Returns:
        14개 특성 dict.

    Raises:
        ValueError: x 비어있음.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) == 0:
        raise ValueError("empty signal")
    return {
        "mean": _safe_float(x.mean()),
        "std": _safe_float(x.std()),
        "min": _safe_float(x.min()),
        "max": _safe_float(x.max()),
        "median": _safe_float(np.median(x)),
        "iqr": _safe_float(np.percentile(x, 75) - np.percentile(x, 25)),
        "mean_above_q75": mean_above_quantile(x, q=0.75),
        "long_run_above_mean": long_run_above_mean(x),
        "n_peaks": number_peaks(x, support=3),
        "first_above_mean": first_index_above_mean(x),
        "pct_below_zero": percentage_below_zero(x),
        "total_variation": absolute_sum_of_changes(x),
        "mean_abs_change": mean_absolute_change(x),
        "acf_lag1": autocorrelation_lag1(x),
        "trend_slope": trend_slope(x),
        "crest_factor": crest_factor(x),
        "shannon_entropy": shannon_entropy(x, bins=10),
        "zero_crossing_rate": zero_crossing_rate(x),
    }


__all__ = [
    "extract_features",
    "mean_above_quantile",
    "long_run_above_mean",
    "number_peaks",
    "first_index_above_mean",
    "percentage_below_zero",
    "absolute_sum_of_changes",
    "mean_absolute_change",
    "autocorrelation_lag1",
    "trend_slope",
    "crest_factor",
    "shannon_entropy",
    "zero_crossing_rate",
]
