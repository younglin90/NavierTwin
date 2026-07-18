"""분위수 / IQR / 백분위 / 박스플롯 통계 후처리.

CFD 시계열에서 강건한(outlier에 덜 민감한) 통계량 추출. 평균/표준편차
대신 중앙값/IQR이 더 안정적인 경우 (heavy-tail 분포, 측정 spike 존재).

상용 툴 대응:
    - Tecplot 360: Box-and-whisker plots
    - Ansys CFD-Post: Statistics → Percentile
    - MATLAB: prctile, iqr, boxplot

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(1000)
    >>> from naviertwin.core.flow_analysis.quantile_stats import box_stats
    >>> b = box_stats(x)
    >>> "median" in b and "iqr" in b
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def percentile(
    x: NDArray[np.float64],
    p: float | NDArray[np.float64],
    axis: int | None = None,
) -> float | NDArray[np.float64]:
    """백분위수. p ∈ [0, 100].

    Args:
        x: 데이터.
        p: 백분위 (스칼라 또는 배열).
        axis: 계산 축.

    Returns:
        백분위 값.

    Raises:
        ValueError: p 범위 오류.
    """
    p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64))
    if np.any(p_arr < 0) or np.any(p_arr > 100):
        raise ValueError("percentile must be in [0, 100]")
    return np.percentile(x, p, axis=axis)


def quartiles(
    x: NDArray[np.float64], axis: int | None = None,
) -> tuple[float, float, float]:
    """Q1, Q2(median), Q3 반환.

    Args:
        x: 데이터.
        axis: 계산 축.

    Returns:
        (Q1, Q2, Q3).
    """
    q1, q2, q3 = np.percentile(x, [25, 50, 75], axis=axis)
    return q1, q2, q3


def iqr(x: NDArray[np.float64], axis: int | None = None) -> float | NDArray[np.float64]:
    """Interquartile range Q3 - Q1.

    Args:
        x: 데이터.
        axis: 계산 축.

    Returns:
        IQR.
    """
    q1, _, q3 = quartiles(x, axis=axis)
    return q3 - q1


def box_stats(
    x: NDArray[np.float64],
    whisker_factor: float = 1.5,
) -> dict[str, float | NDArray[np.float64]]:
    """Tukey 박스플롯 통계량 dict.

    Components:
        median, Q1, Q3, IQR, whisker_low, whisker_high, n_outliers

    Args:
        x: (N,) 데이터.
        whisker_factor: 위스커 길이 (Tukey: 1.5).

    Returns:
        통계량 dict.

    Raises:
        ValueError: whisker_factor < 0.
    """
    if whisker_factor < 0:
        raise ValueError(
            f"whisker_factor must be >= 0, got {whisker_factor}"
        )
    x = np.asarray(x, dtype=np.float64).ravel()
    q1, med, q3 = quartiles(x)
    iqr_val = q3 - q1
    lo_fence = q1 - whisker_factor * iqr_val
    hi_fence = q3 + whisker_factor * iqr_val

    in_range = x[(x >= lo_fence) & (x <= hi_fence)]
    if len(in_range) > 0:
        whisker_low = float(in_range.min())
        whisker_high = float(in_range.max())
    else:
        whisker_low = float(x.min())
        whisker_high = float(x.max())

    n_out = int(np.sum((x < lo_fence) | (x > hi_fence)))

    return {
        "median": float(med),
        "Q1": float(q1),
        "Q3": float(q3),
        "iqr": float(iqr_val),
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
        "n_outliers": n_out,
        "min": float(x.min()),
        "max": float(x.max()),
    }


def empirical_cdf(
    x: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """경험적 누적 분포 함수.

    Args:
        x: 데이터.

    Returns:
        (sorted_x, F(x)) — F(x) = (rank+1) / (N+1).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    s = np.sort(x)
    n = len(s)
    F = (np.arange(1, n + 1)) / (n + 1)
    return s, F


def outliers_iqr(
    x: NDArray[np.float64], whisker_factor: float = 1.5,
) -> NDArray[np.bool_]:
    """IQR 기반 이상치 탐지 마스크.

    Args:
        x: 1D 데이터.
        whisker_factor: 임계값 배수 (기본 Tukey 1.5).

    Returns:
        (N,) 부울 마스크 (True = 이상치).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    q1, _, q3 = quartiles(x)
    iqr_val = q3 - q1
    lo = q1 - whisker_factor * iqr_val
    hi = q3 + whisker_factor * iqr_val
    return (x < lo) | (x > hi)


def trimmed_mean(
    x: NDArray[np.float64],
    trim_fraction: float = 0.1,
) -> float:
    """양 끝 trim_fraction 비율 제거 후 평균.

    Args:
        x: 데이터.
        trim_fraction: 0~0.5. 양 끝에서 제거할 비율.

    Returns:
        Trimmed mean.

    Raises:
        ValueError: trim_fraction 범위 오류.
    """
    if not (0 <= trim_fraction < 0.5):
        raise ValueError(
            f"trim_fraction must be in [0, 0.5), got {trim_fraction}"
        )
    x = np.asarray(x, dtype=np.float64).ravel()
    s = np.sort(x)
    n = len(s)
    cut = int(n * trim_fraction)
    if cut * 2 >= n:
        return float(np.median(s))
    trimmed = s[cut : n - cut]
    return float(trimmed.mean())


def winsorized_mean(
    x: NDArray[np.float64],
    limits: tuple[float, float] = (0.05, 0.05),
) -> float:
    """Winsorize 후 평균: 양 끝의 비율을 임계값으로 클립.

    Args:
        x: 데이터.
        limits: (low_frac, high_frac) ∈ [0, 0.5).

    Returns:
        Winsorized mean.

    Raises:
        ValueError: limits 범위 오류.
    """
    lo, hi = limits
    if not (0 <= lo < 0.5) or not (0 <= hi < 0.5):
        raise ValueError(f"limits must each be in [0, 0.5), got {limits}")
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    n_lo = int(n * lo)
    n_hi = int(n * hi)
    s = np.sort(x).copy()
    if n_lo > 0:
        s[:n_lo] = s[n_lo]
    if n_hi > 0:
        s[-n_hi:] = s[-n_hi - 1]
    return float(s.mean())


__all__ = [
    "percentile",
    "quartiles",
    "iqr",
    "box_stats",
    "empirical_cdf",
    "outliers_iqr",
    "trimmed_mean",
    "winsorized_mean",
]
