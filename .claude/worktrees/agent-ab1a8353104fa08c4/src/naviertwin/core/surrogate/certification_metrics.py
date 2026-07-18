"""Surrogate 모델 인증 메트릭 — CV-RMSE, NRMSE, PICP, 신뢰도 캘리브레이션.

ML/Kriging/RBF 회귀의 예측 품질을 정량화. PICP/MPIW로 예측 구간 커버리지
검증, calibration plot으로 예측 분위와 실제 분위의 일치도 평가.

상용 툴 대응:
    - SMT toolbox: leave_one_out_validation, k_fold
    - SciKit-Learn: cross_val_score, mean_squared_error
    - 학술: Khosravi et al., "Lower upper bound estimation method for
      construction of neural network-based prediction intervals", IEEE TNN 2011.

Examples:
    >>> import numpy as np
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
    >>> y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    >>> from naviertwin.core.surrogate.certification_metrics import (
    ...     normalized_rmse
    ... )
    >>> nrmse = normalized_rmse(y_true, y_pred)
    >>> nrmse < 0.2
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def rmse(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Root Mean Square Error."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true/y_pred shape mismatch: {yt.shape} vs {yp.shape}"
        )
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def normalized_rmse(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    norm: str = "range",
) -> float:
    """Normalized RMSE — RMSE / scale.

    Args:
        y_true, y_pred: 동일 형상.
        norm: "range" (max-min), "mean", "iqr", "std".

    Returns:
        NRMSE (양수, 작을수록 좋음).

    Raises:
        ValueError: norm 옵션 오류 또는 scale = 0.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    err = rmse(y_true, y_pred)
    if norm == "range":
        scale = float(yt.max() - yt.min())
    elif norm == "mean":
        scale = float(np.abs(yt.mean()))
    elif norm == "iqr":
        scale = float(np.percentile(yt, 75) - np.percentile(yt, 25))
    elif norm == "std":
        scale = float(yt.std())
    else:
        raise ValueError(
            f"norm must be 'range'/'mean'/'iqr'/'std', got '{norm}'"
        )
    if scale < 1e-30:
        raise ValueError(f"scale ({norm}) is zero — cannot normalize")
    return err / scale


def cv_rmse(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """Coefficient of Variation of RMSE = RMSE / mean(y_true).

    Args:
        y_true, y_pred: 동일 형상.

    Returns:
        CV-RMSE.
    """
    return normalized_rmse(y_true, y_pred, norm="mean")


def picp(
    y_true: NDArray[np.float64],
    y_lower: NDArray[np.float64],
    y_upper: NDArray[np.float64],
) -> float:
    """Prediction Interval Coverage Probability — y_true ∈ [y_lo, y_hi]의 비율.

    이상적으로 P(1-α) 이상이어야 (예: 95% PI는 PICP ≥ 0.95).

    Args:
        y_true: 실제 값.
        y_lower: 예측 구간 하한.
        y_upper: 예측 구간 상한.

    Returns:
        PICP ∈ [0, 1].

    Raises:
        ValueError: 형상 불일치.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yl = np.asarray(y_lower, dtype=np.float64).ravel()
    yu = np.asarray(y_upper, dtype=np.float64).ravel()
    if yt.shape != yl.shape or yt.shape != yu.shape:
        raise ValueError(
            f"shape mismatch: {yt.shape}, {yl.shape}, {yu.shape}"
        )
    in_range = (yt >= yl) & (yt <= yu)
    return float(np.mean(in_range))


def mpiw(
    y_lower: NDArray[np.float64],
    y_upper: NDArray[np.float64],
    normalize_by_range: NDArray[np.float64] | None = None,
) -> float:
    """Mean Prediction Interval Width — 평균 구간 폭 (좁을수록 좋음).

    Args:
        y_lower, y_upper: 구간.
        normalize_by_range: y_true의 max-min으로 정규화 (옵션).

    Returns:
        MPIW (또는 정규화된 NMPIW).
    """
    yl = np.asarray(y_lower, dtype=np.float64).ravel()
    yu = np.asarray(y_upper, dtype=np.float64).ravel()
    width = np.mean(yu - yl)
    if normalize_by_range is not None:
        ref = np.asarray(normalize_by_range, dtype=np.float64).ravel()
        scale = float(ref.max() - ref.min())
        if scale > 1e-30:
            return float(width / scale)
    return float(width)


def coverage_width_criterion(
    y_true: NDArray[np.float64],
    y_lower: NDArray[np.float64],
    y_upper: NDArray[np.float64],
    target_coverage: float = 0.95,
    eta: float = 50.0,
) -> float:
    """CWC = NMPIW · (1 + γ exp(-η (PICP - μ))).

    Khosravi et al. 2011: 좁고 정확한 PI는 작은 CWC.

    Args:
        y_true: 실제 값.
        y_lower, y_upper: 구간.
        target_coverage: 목표 커버리지 μ.
        eta: 패널티 강도.

    Returns:
        CWC (작을수록 좋음).
    """
    pcp = picp(y_true, y_lower, y_upper)
    mpw = mpiw(y_lower, y_upper, normalize_by_range=y_true)
    if pcp >= target_coverage:
        gamma = 0.0
    else:
        gamma = 1.0
    return float(mpw * (1.0 + gamma * np.exp(-eta * (pcp - target_coverage))))


def calibration_curve(
    y_true: NDArray[np.float64],
    y_mean: NDArray[np.float64],
    y_std: NDArray[np.float64],
    n_quantiles: int = 11,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """캘리브레이션 곡선 — 예측 분위 vs 실제 분위.

    Args:
        y_true: 실제.
        y_mean, y_std: 예측 평균과 표준편차.
        n_quantiles: 평가 분위 수.

    Returns:
        (predicted_q, observed_q). 완벽 캘리브레이션은 y=x.

    Raises:
        ValueError: 형상 불일치.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    ym = np.asarray(y_mean, dtype=np.float64).ravel()
    ys = np.asarray(y_std, dtype=np.float64).ravel()
    if yt.shape != ym.shape or yt.shape != ys.shape:
        raise ValueError("shape mismatch among y_true/y_mean/y_std")
    if np.any(ys < 0):
        raise ValueError("y_std must be non-negative")

    # 표준화 잔차
    z = (yt - ym) / np.maximum(ys, 1e-30)
    # 정규 CDF (오차 함수)
    from math import erf, sqrt

    scale = sqrt(2.0)
    F = np.fromiter(
        map(lambda zi: 0.5 * (1.0 + erf(float(zi) / scale)), z),
        dtype=np.float64,
        count=z.size,
    )

    quantiles = np.linspace(0.0, 1.0, n_quantiles)
    observed = (F[:, None] <= quantiles[None, :]).mean(axis=0)
    return quantiles, observed


def expected_calibration_error(
    y_true: NDArray[np.float64],
    y_mean: NDArray[np.float64],
    y_std: NDArray[np.float64],
    n_quantiles: int = 11,
) -> float:
    """ECE — predicted vs observed 분위 차이의 평균.

    Args:
        y_true, y_mean, y_std: 실제, 예측 평균, 표준편차.
        n_quantiles: 분위 수.

    Returns:
        ECE ≥ 0 (작을수록 좋음).
    """
    pred, obs = calibration_curve(y_true, y_mean, y_std, n_quantiles)
    return float(np.mean(np.abs(pred - obs)))


def r2_score(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """결정 계수 R² ∈ (-∞, 1]. 1 = 완벽."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        raise ValueError("shape mismatch")
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot < 1e-30:
        return float("nan")
    return 1.0 - ss_res / ss_tot


__all__ = [
    "rmse",
    "normalized_rmse",
    "cv_rmse",
    "picp",
    "mpiw",
    "coverage_width_criterion",
    "calibration_curve",
    "expected_calibration_error",
    "r2_score",
]
