"""인과성 분석 — Pearson 상관, Granger causality.

Granger: X 가 Y 를 Granger-cause 한다 ↔ X 의 과거가 Y 의 과거만으로
예측한 Y 의 잔차를 유의하게 줄인다 (F-test).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.sensitivity.causal_analysis import (
    ...     correlation_matrix, granger_causality,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(200)
    >>> y = np.roll(x, 3) + 0.1 * rng.standard_normal(200)  # x 가 y 를 선행
    >>> p = granger_causality(x, y, max_lag=5)
    >>> p[0] < p[3]  # lag 3 에서 p 값이 작아야 (x→y 신호)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def correlation_matrix(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pearson 상관행렬 (열 방향 변수).

    Args:
        X: (N, d).

    Returns:
        (d, d) 대칭 행렬.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X (N, d) 2D 필요: {X.shape}")
    Xc = X - X.mean(axis=0, keepdims=True)
    std = Xc.std(axis=0, ddof=1) + 1e-30
    Xn = Xc / std
    return (Xn.T @ Xn) / (X.shape[0] - 1)


def granger_causality(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    max_lag: int = 5,
) -> NDArray[np.float64]:
    """X → Y 에 대한 각 lag 별 F-test p-value 배열 반환.

    Args:
        x, y: 길이 N 의 1D 시계열.
        max_lag: 최대 지연 수.

    Returns:
        p-values shape (max_lag,). 작을수록 x → y 인과 강함.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"x, y 길이 불일치: {x.size} vs {y.size}")

    try:
        from scipy.stats import f as f_dist
    except ImportError as exc:
        raise RuntimeError("scipy 필요") from exc

    N = x.size
    p_values = np.zeros(max_lag)

    for L in range(1, max_lag + 1):
        if N - L < L + 2:
            p_values[L - 1] = 1.0
            continue

        # Restricted model: y_t ~ y_{t-1..t-L}
        Y = y[L:]
        X_r = np.column_stack([y[L - i - 1 : N - i - 1] for i in range(L)])
        X_r = np.column_stack([np.ones(N - L), X_r])
        beta_r, *_ = np.linalg.lstsq(X_r, Y, rcond=None)
        res_r = Y - X_r @ beta_r
        rss_r = float(np.sum(res_r ** 2))

        # Unrestricted: y_t ~ y_{t-1..t-L} + x_{t-1..t-L}
        X_x = np.column_stack([x[L - i - 1 : N - i - 1] for i in range(L)])
        X_u = np.column_stack([X_r, X_x])
        beta_u, *_ = np.linalg.lstsq(X_u, Y, rcond=None)
        res_u = Y - X_u @ beta_u
        rss_u = float(np.sum(res_u ** 2))

        df_num = L
        df_den = (N - L) - (2 * L + 1)
        if df_den <= 0 or rss_u <= 0:
            p_values[L - 1] = 1.0
            continue

        f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
        p_values[L - 1] = 1.0 - float(f_dist.cdf(f_stat, df_num, df_den))

    logger.info(
        "granger_causality: p_values(lag 1..%d) = %s",
        max_lag, np.round(p_values, 4).tolist(),
    )
    return p_values


__all__ = ["correlation_matrix", "granger_causality"]
