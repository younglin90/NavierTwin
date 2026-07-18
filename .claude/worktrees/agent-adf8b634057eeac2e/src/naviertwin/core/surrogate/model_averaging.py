"""Bayesian Model Averaging + 앙상블 가중 — 다중 surrogate 결합.

여러 surrogate 모델 (POD-RBF, POD-Kriging, FNO, ...)의 예측을 가중 평균.
가중치는 검증 오차 기반 또는 BIC/Bayes 기반.

상용 툴 대응:
    - SciKit-Learn: VotingRegressor, StackingRegressor
    - 학술: Hoeting et al., "Bayesian Model Averaging: A Tutorial",
      Statistical Science, 1999.

Examples:
    >>> import numpy as np
    >>> # 두 모델 예측 평균
    >>> p1 = np.array([1.0, 2.0, 3.0])
    >>> p2 = np.array([1.5, 2.5, 2.5])
    >>> from naviertwin.core.surrogate.model_averaging import equal_weight_average
    >>> avg = equal_weight_average([p1, p2])
    >>> avg.tolist()
    [1.25, 2.25, 2.75]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _stack_predictions(predictions: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    return np.stack(tuple(map(lambda p: np.asarray(p, dtype=np.float64), predictions)))


def equal_weight_average(
    predictions: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """단순 산술 평균.

    Args:
        predictions: 모델 예측 리스트 (모두 같은 형상).

    Returns:
        평균 예측.

    Raises:
        ValueError: 빈 리스트 또는 형상 불일치.
    """
    if len(predictions) == 0:
        raise ValueError("predictions list is empty")
    arr = _stack_predictions(predictions)
    return arr.mean(axis=0)


def weighted_average(
    predictions: list[NDArray[np.float64]],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """가중 평균 ŷ = Σ w_k ŷ_k.

    Args:
        predictions: 모델 예측 리스트.
        weights: (M,) 가중치. 합 = 1로 자동 정규화.

    Returns:
        가중 평균.

    Raises:
        ValueError: 형상 불일치.
    """
    M = len(predictions)
    if M == 0:
        raise ValueError("predictions list is empty")
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != (M,):
        raise ValueError(f"weights length {w.shape[0]} != n_models {M}")
    w_sum = w.sum()
    if w_sum < 1e-30:
        raise ValueError("weights sum to zero")
    w = w / w_sum

    arr = _stack_predictions(predictions)
    return np.einsum("i,i...->...", w, arr)


def cv_error_weights(
    cv_errors: NDArray[np.float64],
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """CV 오차 역수 기반 가중 (작은 오차 → 큰 가중치).

    w_k = (1/e_k²) / Σ (1/e_j²).

    Args:
        cv_errors: (M,) 각 모델의 CV 오차.
        eps: 0 분모 방지.

    Returns:
        (M,) 정규화된 가중치.

    Raises:
        ValueError: 음수 오차.
    """
    e = np.asarray(cv_errors, dtype=np.float64).ravel()
    if np.any(e < 0):
        raise ValueError("CV errors must be non-negative")
    inv_sq = 1.0 / (e ** 2 + eps)
    return inv_sq / inv_sq.sum()


def bic_weights(
    bic_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """BIC-기반 모델 가중 (Bayesian Model Averaging).

    w_k ∝ exp(-½ ΔBIC_k), where ΔBIC_k = BIC_k - min(BIC).

    Args:
        bic_values: (M,) 각 모델의 BIC.

    Returns:
        (M,) 정규화된 사후 확률.
    """
    b = np.asarray(bic_values, dtype=np.float64).ravel()
    delta = b - b.min()
    raw = np.exp(-0.5 * delta)
    return raw / raw.sum()


def stacking_least_squares(
    predictions: NDArray[np.float64],
    y_true: NDArray[np.float64],
    nonnegative: bool = True,
) -> NDArray[np.float64]:
    """스태킹: ŷ_stack = Σ w_k p_k 의 가중치를 최소자승으로 추정.

    Args:
        predictions: (n, M) 각 모델의 예측 (열 = 모델).
        y_true: (n,) 실제 값.
        nonnegative: True면 w_k ≥ 0 제약 (간단한 NNLS 근사).

    Returns:
        (M,) 가중치 (sum = 1로 정규화).

    Raises:
        ValueError: 형상 불일치.
    """
    P = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64).ravel()
    if P.ndim != 2:
        raise ValueError(f"predictions must be 2D, got {P.shape}")
    if P.shape[0] != y.shape[0]:
        raise ValueError(
            f"predictions rows {P.shape[0]} != y_true {y.shape[0]}"
        )

    # OLS
    w, _, _, _ = np.linalg.lstsq(P, y, rcond=None)

    if nonnegative:
        w = np.maximum(w, 0.0)

    s = w.sum()
    if abs(s) < 1e-30:
        # 폴백: 균등
        return np.ones(P.shape[1]) / P.shape[1]
    return w / s


def ensemble_variance(
    predictions: list[NDArray[np.float64]],
    weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """앙상블 예측의 분산 (model uncertainty).

    Var = Σ w_k (ŷ_k - ŷ_avg)².

    Args:
        predictions: 모델 예측 리스트.
        weights: 가중치 (None이면 균등).

    Returns:
        분산 배열.

    Raises:
        ValueError: 빈 리스트.
    """
    M = len(predictions)
    if M == 0:
        raise ValueError("predictions list is empty")
    arr = _stack_predictions(predictions)
    if weights is None:
        w = np.ones(M) / M
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        s = w.sum()
        if s < 1e-30:
            raise ValueError("weights sum to zero")
        w = w / s
    avg = np.einsum("i,i...->...", w, arr)
    diffsq = (arr - avg[None, ...]) ** 2
    return np.einsum("i,i...->...", w, diffsq)


__all__ = [
    "equal_weight_average",
    "weighted_average",
    "cv_error_weights",
    "bic_weights",
    "stacking_least_squares",
    "ensemble_variance",
]
