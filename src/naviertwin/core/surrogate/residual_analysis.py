"""Surrogate 잔차 분석 — Q-Q plot, 잔차 자기상관, leverage, Cook 거리.

학습된 surrogate(Kriging/RBF/NN)의 예측 오차에 패턴이 있는지 진단.
잔차가 정규/IID인지 확인 → 모델 적합 여부 판정.

상용 툴 대응:
    - R: car / lmtest 잔차 진단 패키지
    - SciPy: stats.probplot
    - MATLAB: qqplot, residual_diagnostics
    - 학술: Belsley et al., "Regression Diagnostics", 1980.

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y_true = rng.standard_normal(100)
    >>> y_pred = y_true + 0.1 * rng.standard_normal(100)
    >>> from naviertwin.core.surrogate.residual_analysis import qq_data
    >>> sorted_residuals, normal_quantiles = qq_data(y_true - y_pred)
    >>> sorted_residuals.shape
    (100,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def qq_data(
    residuals: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Q-Q plot 데이터 — 정렬된 잔차 vs 이론 정규 분위.

    Args:
        residuals: (N,) 잔차.

    Returns:
        (sorted_residuals, theoretical_quantiles). 정규 분포 → 직선.

    Raises:
        ValueError: residuals 길이 < 2.
    """
    r = np.asarray(residuals, dtype=np.float64).ravel()
    n = len(r)
    if n < 2:
        raise ValueError(f"need at least 2 residuals, got {n}")

    sorted_r = np.sort(r)
    # Filliben/blom 표본 분위
    p = (np.arange(1, n + 1) - 0.5) / n

    # Φ⁻¹ 근사
    from math import sqrt

    def _norm_inv(q: float) -> float:
        # Beasley-Springer-Moro
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        p_low = 0.02425
        p_high = 1 - p_low
        if q < p_low:
            qq = sqrt(-2.0 * np.log(q))
            return (((((c[0] * qq + c[1]) * qq + c[2]) * qq + c[3]) * qq + c[4]) * qq + c[5]) / (
                (((d[0] * qq + d[1]) * qq + d[2]) * qq + d[3]) * qq + 1.0
            )
        if q <= p_high:
            qd = q - 0.5
            r2 = qd * qd
            return (((((a[0] * r2 + a[1]) * r2 + a[2]) * r2 + a[3]) * r2 + a[4]) * r2 + a[5]) * qd / (
                ((((b[0] * r2 + b[1]) * r2 + b[2]) * r2 + b[3]) * r2 + b[4]) * r2 + 1.0
            )
        qq = sqrt(-2.0 * np.log(1.0 - q))
        return -(((((c[0] * qq + c[1]) * qq + c[2]) * qq + c[3]) * qq + c[4]) * qq + c[5]) / (
            (((d[0] * qq + d[1]) * qq + d[2]) * qq + d[3]) * qq + 1.0
        )

    theoretical = np.fromiter(map(_norm_inv, p), dtype=np.float64, count=p.size)
    return sorted_r, theoretical


def residual_autocorrelation(
    residuals: NDArray[np.float64],
    max_lag: int | None = None,
) -> NDArray[np.float64]:
    """잔차 자기상관 — 시간/공간 패턴 검출.

    Args:
        residuals: (N,) 잔차.
        max_lag: 최대 랙. None이면 N//4.

    Returns:
        (max_lag+1,) 자기상관 (R[0]=1).
    """
    r = np.asarray(residuals, dtype=np.float64).ravel()
    n = len(r)
    if n < 2:
        return np.array([1.0])
    if max_lag is None:
        max_lag = n // 4
    max_lag = min(max_lag, n - 1)

    rp = r - r.mean()
    var = float(np.dot(rp, rp)) + 1e-30
    out = np.correlate(rp, rp, mode="full")[n - 1 : n + max_lag] / var
    out[0] = 1.0
    return out


def durbin_watson(
    residuals: NDArray[np.float64],
) -> float:
    """Durbin-Watson 통계량 — 잔차 자기상관 진단.

    DW ∈ [0, 4]: 2 ≈ no autocorrelation, < 2 양의 상관, > 2 음의 상관.

    Args:
        residuals: (N,) 잔차.

    Returns:
        DW 값.

    Raises:
        ValueError: residuals 길이 < 2.
    """
    r = np.asarray(residuals, dtype=np.float64).ravel()
    if len(r) < 2:
        raise ValueError(f"need at least 2 residuals, got {len(r)}")
    diff = np.diff(r)
    num = float(np.sum(diff ** 2))
    den = float(np.sum(r ** 2)) + 1e-30
    return num / den


def leverage_scores(
    X: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Hat 행렬 H = X(XᵀX)⁻¹Xᵀ의 대각 성분 (각 점의 leverage).

    h_i ∈ [0, 1]; 평균 = p/n where p = 변수 수, n = 표본 수.
    h_i > 2p/n 인 점은 고-leverage (모델에 큰 영향).

    Args:
        X: (n, p) 회귀 디자인 행렬.

    Returns:
        (n,) leverage 배열.

    Raises:
        ValueError: X가 2D 아님.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    # H = X (XᵀX)⁻¹ Xᵀ → 대각 = sum(X_i (XᵀX)⁻¹ X_i)
    XtX = X.T @ X + 1e-12 * np.eye(X.shape[1])
    inv = np.linalg.inv(XtX)
    diag = np.einsum("ij,jk,ik->i", X, inv, X)
    return np.clip(diag, 0.0, 1.0)


def cooks_distance(
    residuals: NDArray[np.float64],
    leverage: NDArray[np.float64],
    mse: float,
    p: int,
) -> NDArray[np.float64]:
    """Cook 거리 D_i = (e_i² / (p · MSE)) · (h_ii / (1 - h_ii)²).

    영향력 측정: D_i > 1 인 점은 고영향.

    Args:
        residuals: (n,) 잔차.
        leverage: (n,) hat 대각.
        mse: 잔차 평균 제곱.
        p: 회귀 변수 수.

    Returns:
        (n,) Cook 거리.

    Raises:
        ValueError: 매개변수 오류.
    """
    r = np.asarray(residuals, dtype=np.float64).ravel()
    h = np.asarray(leverage, dtype=np.float64).ravel()
    if r.shape != h.shape:
        raise ValueError("residuals/leverage shape mismatch")
    if p <= 0 or mse <= 0:
        raise ValueError(f"p > 0 and mse > 0 required, got p={p}, mse={mse}")

    one_minus_h = np.clip(1.0 - h, 1e-30, 1.0)
    return (r ** 2 / (p * mse)) * (h / (one_minus_h ** 2))


def shapiro_normality_diagnostic(
    residuals: NDArray[np.float64],
) -> dict[str, float]:
    """잔차 정규성 진단 통계량 dict.

    포함: mean, std, skewness, kurtosis (excess), DW.

    Args:
        residuals: 잔차.

    Returns:
        통계량 dict.
    """
    r = np.asarray(residuals, dtype=np.float64).ravel()
    n = len(r)
    if n < 2:
        return {"mean": 0.0, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0, "dw": 2.0}
    mu = r.mean()
    sigma = r.std(ddof=1) + 1e-30
    z = (r - mu) / sigma
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4) - 3.0)
    return {
        "mean": float(mu),
        "std": float(sigma),
        "skewness": skew,
        "kurtosis": kurt,
        "dw": durbin_watson(r),
    }


__all__ = [
    "qq_data",
    "residual_autocorrelation",
    "durbin_watson",
    "leverage_scores",
    "cooks_distance",
    "shapiro_normality_diagnostic",
]
