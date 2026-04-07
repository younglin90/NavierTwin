"""검증 지표 계산.

CFD/AI 예측 결과의 정확도를 평가하기 위한 다양한 지표 함수를 제공한다.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def rmse(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Root Mean Squared Error (RMSE).

    Args:
        y_true: 실제값. 임의 shape.
        y_pred: 예측값. y_true와 동일한 shape.

    Returns:
        RMSE 값 (float, 0 이상).

    Examples:
        >>> import numpy as np
        >>> from naviertwin.core.validation.metrics import rmse
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([1.1, 1.9, 3.2])
        >>> rmse(y_true, y_pred)
        0.14...
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """결정계수 R² (Coefficient of Determination).

    R² = 1 - SS_res / SS_tot

    1.0이 완벽한 예측, 0.0은 평균값 예측과 동등, 음수는 평균보다 나쁜 예측.

    Args:
        y_true: 실제값. 임의 shape.
        y_pred: 예측값. y_true와 동일한 shape.

    Returns:
        R² 점수 (float).

    Examples:
        >>> import numpy as np
        >>> from naviertwin.core.validation.metrics import r2_score
        >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
        >>> y_pred = np.array([1.1, 1.9, 3.1, 3.9])
        >>> r2_score(y_true, y_pred)
        0.99...
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def relative_l2_error(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> float:
    """상대 L2 노름 오차.

    ||y_pred - y_true||₂ / ||y_true||₂

    y_true의 L2 노름이 0이면 절대 오차를 반환한다.

    Args:
        y_true: 실제값. 임의 shape.
        y_pred: 예측값. y_true와 동일한 shape.

    Returns:
        상대 L2 오차 (float, 0 이상).

    Examples:
        >>> import numpy as np
        >>> from naviertwin.core.validation.metrics import relative_l2_error
        >>> y_true = np.ones(5)
        >>> y_pred = np.ones(5) * 1.1
        >>> relative_l2_error(y_true, y_pred)
        0.1...
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    norm_true = float(np.linalg.norm(y_true))
    norm_diff = float(np.linalg.norm(y_pred - y_true))

    if norm_true == 0.0:
        logger.warning(
            "relative_l2_error: y_true의 L2 노름이 0입니다. 절대 오차를 반환합니다."
        )
        return norm_diff
    return norm_diff / norm_true


def max_error(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> float:
    """최대 절대 오차 (Maximum Absolute Error).

    max(|y_pred - y_true|)

    Args:
        y_true: 실제값. 임의 shape.
        y_pred: 예측값. y_true와 동일한 shape.

    Returns:
        최대 절대 오차 (float, 0 이상).

    Examples:
        >>> import numpy as np
        >>> from naviertwin.core.validation.metrics import max_error
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([1.5, 2.0, 3.0])
        >>> max_error(y_true, y_pred)
        0.5
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.max(np.abs(y_true - y_pred)))


def compute_all_metrics(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> dict[str, float]:
    """모든 검증 지표를 딕셔너리로 반환한다.

    Args:
        y_true: 실제값. 임의 shape.
        y_pred: 예측값. y_true와 동일한 shape.

    Returns:
        검증 지표 딕셔너리:
            - "rmse": Root Mean Squared Error
            - "r2": 결정계수 R²
            - "relative_l2": 상대 L2 노름 오차
            - "max_error": 최대 절대 오차

    Examples:
        >>> import numpy as np
        >>> from naviertwin.core.validation.metrics import compute_all_metrics
        >>> rng = np.random.default_rng(0)
        >>> y_true = rng.standard_normal(100)
        >>> y_pred = y_true + 0.1 * rng.standard_normal(100)
        >>> metrics = compute_all_metrics(y_true, y_pred)
        >>> set(metrics.keys()) == {"rmse", "r2", "relative_l2", "max_error"}
        True
    """
    result = {
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "relative_l2": relative_l2_error(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
    }
    logger.debug(
        "compute_all_metrics: rmse=%.6f, r2=%.6f, rel_l2=%.6f, max_err=%.6f",
        result["rmse"],
        result["r2"],
        result["relative_l2"],
        result["max_error"],
    )
    return result
