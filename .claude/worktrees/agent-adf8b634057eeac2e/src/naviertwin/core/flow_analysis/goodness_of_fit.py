"""분포 적합도 검정 — Kolmogorov-Smirnov / Anderson-Darling / χ².

CFD 시계열의 통계 분포가 가우시안/지수/대수정규/사용자 정의 분포와
얼마나 일치하는지 검정. UQ 모델 검증, 사후 잔차 분석에 사용.

상용 툴 대응:
    - MATLAB Statistics Toolbox: kstest, adtest, chi2gof
    - SciPy: stats.kstest, stats.anderson, stats.chisquare
    - 학술: Press et al., "Numerical Recipes" Ch. 14.

References:
    Anderson, T.W. & Darling, D.A., "Asymptotic theory of certain
    'goodness of fit' criteria based on stochastic processes",
    Ann. Math. Statist., 1952.

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(1000)
    >>> from naviertwin.core.flow_analysis.goodness_of_fit import ks_test_normal
    >>> stat, p = ks_test_normal(x)
    >>> p > 0.05  # 정규분포 → 적합도 높음
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.utils.logger import get_logger

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")

logger = get_logger(__name__)


def _kolmogorov_pvalue(D: float, n: int) -> float:
    """KS 통계량 D와 표본 크기 n에서 점진적 p-value (Marsaglia 1956 근사)."""
    return float(_kernels.kolmogorov_pvalue(float(D), float(n)))


def ks_test_against_cdf(
    x: NDArray[np.float64],
    cdf: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> tuple[float, float]:
    """1-sample Kolmogorov-Smirnov 검정: D = max |F_n(x) - F(x)|.

    Args:
        x: (N,) 표본.
        cdf: 비교할 누적 분포 함수.

    Returns:
        (D_stat, p_value).

    Raises:
        ValueError: x가 비어있는 경우.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 2:
        raise ValueError(f"need at least 2 samples, got {n}")

    s = np.sort(x)
    F_emp_lo = np.arange(0, n) / n
    F_emp_hi = np.arange(1, n + 1) / n
    F_th = cdf(s)
    F_th = np.clip(F_th, 0.0, 1.0)

    D_plus = np.max(F_emp_hi - F_th)
    D_minus = np.max(F_th - F_emp_lo)
    D = float(max(D_plus, D_minus))
    p = _kolmogorov_pvalue(D, n)
    return D, p


def normal_cdf(x: NDArray[np.float64], mu: float = 0.0, sigma: float = 1.0) -> NDArray[np.float64]:
    """정규 분포 CDF (오차 함수 기반)."""
    z = (np.asarray(x) - mu) / max(sigma, 1e-30)
    return _kernels.norm_cdf(np.asarray(z, dtype=np.float64))


def ks_test_normal(
    x: NDArray[np.float64],
    mu: float | None = None,
    sigma: float | None = None,
) -> tuple[float, float]:
    """KS 검정 vs 정규 분포. mu/sigma None이면 표본 통계 사용.

    Args:
        x: (N,) 표본.
        mu: 평균 (None이면 표본 평균).
        sigma: 표준편차 (None이면 표본 표준편차).

    Returns:
        (D, p_value).

    Raises:
        ValueError: 표본 부족.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) < 2:
        raise ValueError(f"need at least 2 samples, got {len(x)}")
    if mu is None:
        mu = float(x.mean())
    if sigma is None:
        sigma = float(x.std(ddof=1))
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    return ks_test_against_cdf(x, lambda v: normal_cdf(v, mu, sigma))


def ks_test_two_sample(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[float, float]:
    """2-sample KS 검정 — 두 표본이 같은 분포에서 나왔는지 검정.

    Args:
        x, y: 두 표본.

    Returns:
        (D, p_value).

    Raises:
        ValueError: 표본 부족.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) < 2 or len(y) < 2:
        raise ValueError(
            f"both samples need >= 2 elements, got {len(x)}, {len(y)}"
        )

    data_all = np.concatenate([x, y])
    cdf_x = np.searchsorted(np.sort(x), data_all, side="right") / len(x)
    cdf_y = np.searchsorted(np.sort(y), data_all, side="right") / len(y)
    D = float(np.max(np.abs(cdf_x - cdf_y)))

    n_eff = len(x) * len(y) / (len(x) + len(y))
    p = _kolmogorov_pvalue(D, n_eff)
    return D, p


def anderson_darling_normal(
    x: NDArray[np.float64],
) -> tuple[float, dict[str, float]]:
    """Anderson-Darling 정규성 검정. 임계값 표 반환 (D'Agostino).

    Args:
        x: (N,) 표본.

    Returns:
        (A², critical_values_dict): 임계값 dict {"15%", "10%", "5%", "2.5%", "1%"}.
        A²이 임계값보다 크면 해당 유의수준에서 정규성 기각.

    Raises:
        ValueError: 표본 < 8 또는 σ=0.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 8:
        raise ValueError(f"AD test needs at least 8 samples, got {n}")
    sigma = float(x.std(ddof=1))
    if sigma < 1e-30:
        raise ValueError("standard deviation is zero")
    z = np.sort((x - x.mean()) / sigma)
    F = normal_cdf(z)
    F = np.clip(F, 1e-12, 1.0 - 1e-12)
    i = np.arange(1, n + 1)
    S = np.sum((2 * i - 1) * (np.log(F) + np.log(1.0 - F[::-1])))
    A2 = -n - S / n
    # 표본 크기 보정 (D'Agostino & Stephens 1986)
    A2_adj = A2 * (1.0 + 0.75 / n + 2.25 / n ** 2)

    critical_values = {
        "15%": 0.576,
        "10%": 0.656,
        "5%": 0.787,
        "2.5%": 0.918,
        "1%": 1.092,
    }
    return float(A2_adj), critical_values


def chi_square_test(
    observed: NDArray[np.float64],
    expected: NDArray[np.float64],
    ddof: int = 0,
) -> tuple[float, int]:
    """χ² 적합도 통계량 = Σ (O - E)² / E.

    p-value는 자유도와 함께 사용자가 별도 계산 (chi2.sf).

    Args:
        observed: (k,) 관찰 빈도.
        expected: (k,) 기대 빈도.
        ddof: 자유도 보정.

    Returns:
        (chi2_stat, dof): dof = k - 1 - ddof.

    Raises:
        ValueError: 형상 불일치 또는 expected ≤ 0.
    """
    obs = np.asarray(observed, dtype=np.float64).ravel()
    exp = np.asarray(expected, dtype=np.float64).ravel()
    if obs.shape != exp.shape:
        raise ValueError(
            f"observed/expected shape mismatch: {obs.shape} vs {exp.shape}"
        )
    if np.any(exp <= 0):
        raise ValueError("expected frequencies must be > 0")

    chi2 = float(np.sum((obs - exp) ** 2 / exp))
    dof = max(1, len(obs) - 1 - ddof)
    return chi2, dof


def shapiro_wilk_simplified(
    x: NDArray[np.float64],
) -> tuple[float, float]:
    """단순화된 Shapiro-Wilk 정규성 검정 (W 통계량 + 근사 p).

    표본 8 ≤ n ≤ 50에 적합. Royston 1982 근사 사용.

    Args:
        x: 1D 표본.

    Returns:
        (W, p_approx).

    Raises:
        ValueError: 표본 < 8.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 8:
        raise ValueError(f"need at least 8 samples, got {n}")

    s = np.sort(x)
    mean = s.mean()
    # Blom 근사 m_i
    i = np.arange(1, n + 1)
    probs = (i - 0.375) / (n + 0.25)
    m = -np.sqrt(2.0) * np.fromiter(
        map(_inverse_norm_cdf, probs),
        dtype=np.float64,
        count=n,
    )
    m = -m  # ascending → expected order stats
    m_norm = float(np.sqrt((m * m).sum()))
    a = m / max(m_norm, 1e-30)

    # symmetric weighting
    W_num = float((a * s).sum()) ** 2
    W_den = float(np.sum((s - mean) ** 2))
    if W_den < 1e-30:
        return 1.0, 1.0
    W = W_num / W_den

    # Royston 근사 p (간단)
    # ln(1 - W) → 정규
    if W >= 1.0 - 1e-12:
        return float(W), 1.0
    g = np.log(1.0 - W)
    mu = -1.5861 - 0.31082 * np.log(n) - 0.083751 * np.log(n) ** 2 + 0.0038915 * np.log(n) ** 3
    sigma = np.exp(-0.4803 - 0.082676 * np.log(n) + 0.0030302 * np.log(n) ** 2)
    z = (g - mu) / sigma
    p = 1.0 - normal_cdf(np.array([z]))[0]
    return float(W), float(np.clip(p, 0.0, 1.0))


def _inverse_norm_cdf(p: float) -> float:
    """Beasley-Springer-Moro 근사 (Φ⁻¹)."""
    if p <= 0:
        return -np.inf
    if p >= 1:
        return np.inf
    # rational approximation
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
    if p < p_low:
        q = np.sqrt(-2.0 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    q = np.sqrt(-2.0 * np.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


__all__ = [
    "ks_test_against_cdf",
    "ks_test_normal",
    "ks_test_two_sample",
    "anderson_darling_normal",
    "chi_square_test",
    "shapiro_wilk_simplified",
    "normal_cdf",
]
