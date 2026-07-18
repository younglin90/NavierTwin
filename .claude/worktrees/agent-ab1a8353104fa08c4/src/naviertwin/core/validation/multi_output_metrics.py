"""다중 출력 surrogate 검증 메트릭.

Surrogate가 여러 출력(스칼라 또는 벡터/장)을 동시에 예측할 때 통합 평가.
각 출력 채널의 RMSE 외에도 cross-channel correlation, channel-wise 가중,
top-K 오차 등을 제공.

상용 툴 대응:
    - sklearn: multioutput_regression metrics (mean_absolute_error 등)
    - 학술: Borchani et al., "A Survey on Multi-output Regression",
      WIREs Data Mining and Knowledge Discovery, 2015.

Examples:
    >>> import numpy as np
    >>> y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> y_pred = y_true + 0.1
    >>> from naviertwin.core.validation.multi_output_metrics import (
    ...     channel_rmse, multi_output_r2
    ... )
    >>> channel_rmse(y_true, y_pred).shape
    (2,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required by multi-output metrics")


def channel_rmse(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> NDArray[np.float64]:
    """채널별 RMSE.

    Args:
        y_true: (N, K) 실제 출력.
        y_pred: (N, K) 예측 출력.

    Returns:
        (K,) 채널별 RMSE.

    Raises:
        ValueError: 형상 불일치 또는 2D 아님.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(
            f"shape mismatch: {yt.shape} vs {yp.shape}"
        )
    if yt.ndim != 2:
        raise ValueError(f"y_true must be 2D, got {yt.shape}")
    return np.sqrt(np.mean((yt - yp) ** 2, axis=0))


def channel_relative_error(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> NDArray[np.float64]:
    """채널별 상대 L2 오차 = ‖y - ŷ‖₂ / ‖y‖₂.

    Args:
        y_true, y_pred: (N, K) 형상.

    Returns:
        (K,) 상대 오차.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape or yt.ndim != 2:
        raise ValueError(f"shape error: {yt.shape}, {yp.shape}")
    num = np.linalg.norm(yt - yp, axis=0)
    den = np.linalg.norm(yt, axis=0) + 1e-30
    return num / den


def aggregated_rmse(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> float:
    """가중 통합 RMSE = √(Σ w_k RMSE_k²).

    Args:
        y_true, y_pred: (N, K).
        weights: (K,) 채널 가중치. None이면 균등.

    Returns:
        스칼라 RMSE.
    """
    rmse_k = channel_rmse(y_true, y_pred)
    K = rmse_k.shape[0]
    if weights is None:
        w = np.ones(K) / K
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape != (K,):
            raise ValueError(f"weights shape {w.shape} != ({K},)")
        s = w.sum()
        if s < 1e-30:
            raise ValueError("weights sum to zero")
        w = w / s
    return float(np.sqrt(np.sum(w * rmse_k ** 2)))


def multi_output_r2(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    average: str = "uniform",
) -> float:
    """다중 출력 R² 점수.

    Args:
        y_true, y_pred: (N, K).
        average: "uniform" (각 채널 R²의 평균),
                 "variance_weighted" (분산 가중),
                 "raw" (None — 각 채널 R² 배열 반환).

    Returns:
        스칼라 R² 또는 (K,) 배열.

    Raises:
        ValueError: average 옵션 오류.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape or yt.ndim != 2:
        raise ValueError(f"shape error: {yt.shape}, {yp.shape}")
    r2_per_channel, var_per_channel = _kernels.multi_output_r2_raw(yt, yp)
    r2_per_channel = np.asarray(r2_per_channel, dtype=np.float64)
    var_per_channel = np.asarray(var_per_channel, dtype=np.float64)

    if average == "raw":
        return r2_per_channel  # type: ignore[return-value]
    if average == "uniform":
        return float(np.nanmean(r2_per_channel))
    if average == "variance_weighted":
        valid = ~np.isnan(r2_per_channel) & (var_per_channel > 0)
        if not valid.any():
            return float("nan")
        weights = var_per_channel[valid] / var_per_channel[valid].sum()
        return float(np.sum(weights * r2_per_channel[valid]))
    raise ValueError(
        f"average must be 'uniform'/'variance_weighted'/'raw', got '{average}'"
    )


def cross_channel_correlation(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> NDArray[np.float64]:
    """채널별 (y_true, y_pred) Pearson 상관.

    Args:
        y_true, y_pred: (N, K).

    Returns:
        (K,) ρ ∈ [-1, 1].
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape or yt.ndim != 2:
        raise ValueError(f"shape error: {yt.shape}, {yp.shape}")
    return _kernels.cross_channel_correlation(yt, yp)


def top_k_worst_channels(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    k: int = 3,
) -> NDArray[np.intp]:
    """RMSE 기준 가장 나쁜 K 채널 인덱스.

    Args:
        y_true, y_pred: (N, K).
        k: 반환할 채널 수.

    Returns:
        (k,) 인덱스 (RMSE 큰 순).
    """
    rmse_k = channel_rmse(y_true, y_pred)
    K = rmse_k.shape[0]
    if k <= 0 or k > K:
        raise ValueError(f"k in [1, {K}], got {k}")
    return np.argsort(-rmse_k)[:k].astype(np.intp)


def per_sample_error_norm(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> NDArray[np.float64]:
    """각 표본의 다채널 오차 노름 ‖y_i - ŷ_i‖₂.

    Args:
        y_true, y_pred: (N, K).

    Returns:
        (N,) 표본별 오차.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape or yt.ndim != 2:
        raise ValueError(f"shape error: {yt.shape}, {yp.shape}")
    return np.linalg.norm(yt - yp, axis=1)


__all__ = [
    "channel_rmse",
    "channel_relative_error",
    "aggregated_rmse",
    "multi_output_r2",
    "cross_channel_correlation",
    "top_k_worst_channels",
    "per_sample_error_norm",
]
