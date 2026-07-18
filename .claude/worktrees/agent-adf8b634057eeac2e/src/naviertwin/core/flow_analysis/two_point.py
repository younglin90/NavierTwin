"""두-점 공간 상관 함수 + 적분 길이 스케일.

CFD 시간 시계열에서 공간 두-점 상관 R(r)과 시간 자기상관 R(τ)을
계산하고, 적분 길이/시간 스케일을 추출한다. 난류 후처리의 핵심.

상용 툴 대응:
    - Tecplot 360: Time-Average Spatial Correlation
    - Ansys CFD-Post: Two-point statistics
    - MATLAB: xcorr2

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> u = rng.standard_normal((100, 50))  # (n_t, n_space)
    >>> from naviertwin.core.flow_analysis.two_point import (
    ...     spatial_autocorrelation, integral_length_scale_from_R
    ... )
    >>> r, R = spatial_autocorrelation(u, dx=0.01)
    >>> R[0] > 0.99
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _fluctuation(u: NDArray[np.float64], time_axis: int = 0) -> NDArray[np.float64]:
    return u - u.mean(axis=time_axis, keepdims=True)


def spatial_autocorrelation(
    u: NDArray[np.float64],
    dx: float = 1.0,
    max_lag: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """공간 자기상관 R_uu(r) = <u'(x) u'(x+r)> / <u'²>.

    시간 평균에 대해 동질성을 가정 (정상 흐름).

    Args:
        u: (n_t, n_space) 시간-공간 배열.
        dx: 격자 간격.
        max_lag: 최대 랙 (격자 점 수). None이면 n_space // 2.

    Returns:
        (r, R): r은 거리 배열, R은 상관 (R[0] = 1).

    Raises:
        ValueError: u가 2D 아님.
    """
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError(f"u must be 2D (n_t, n_space), got {u.shape}")
    n_t, n_x = u.shape
    if max_lag is None:
        max_lag = n_x // 2
    max_lag = min(max_lag, n_x - 1)

    up = _fluctuation(u)
    var = np.mean(up * up) + 1e-30

    R = np.zeros(max_lag + 1)
    R[0] = 1.0
    lags = np.arange(1, max_lag + 1)
    R[1:] = np.fromiter(
        map(lambda lag: float(np.mean(up[:, : n_x - lag] * up[:, lag:]) / var), lags),
        dtype=np.float64,
        count=max_lag,
    )

    r_arr = np.arange(max_lag + 1) * dx
    return r_arr, R


def temporal_autocorrelation(
    u: NDArray[np.float64],
    dt: float = 1.0,
    max_lag: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """시간 자기상관 R_uu(τ) = <u'(t) u'(t+τ)> / <u'²>.

    공간 차원이 있으면 각 점에서 별도 계산 후 평균.

    Args:
        u: (n_t,) 또는 (n_t, n_space) 시계열.
        dt: 시간 간격.
        max_lag: 최대 시간 랙. None이면 n_t // 2.

    Returns:
        (tau, R): 시간 랙과 상관.

    Raises:
        ValueError: u 차원 오류.
    """
    u = np.asarray(u, dtype=np.float64)
    if u.ndim == 1:
        u = u[:, None]
    if u.ndim != 2:
        raise ValueError(f"u must be 1D or 2D, got {u.shape}")
    n_t, n_x = u.shape
    if max_lag is None:
        max_lag = n_t // 2
    max_lag = min(max_lag, n_t - 1)

    up = _fluctuation(u)
    lags = np.arange(1, max_lag + 1)

    def _series_autocorrelation(v: NDArray[np.float64]) -> NDArray[np.float64]:
        var = np.mean(v * v) + 1e-30
        values = np.empty(max_lag + 1, dtype=np.float64)
        values[0] = 1.0
        values[1:] = np.fromiter(
            map(lambda tau: float(np.mean(v[: n_t - tau] * v[tau:]) / var), lags),
            dtype=np.float64,
            count=max_lag,
        )
        return values

    R_per_x = np.apply_along_axis(_series_autocorrelation, 0, up)
    R = R_per_x.mean(axis=1)
    tau_arr = np.arange(max_lag + 1) * dt
    return tau_arr, R


def integral_length_scale_from_R(
    r: NDArray[np.float64],
    R: NDArray[np.float64],
) -> float:
    """적분 길이 스케일 L = ∫₀^∞ R(r) dr.

    R이 음수가 되는 첫 번째 점까지 적분 (정상적 사례).

    Args:
        r: 거리 배열 (단조 증가).
        R: 상관 함수.

    Returns:
        L (단위: r과 같음).

    Raises:
        ValueError: r과 R 형상 불일치.
    """
    r = np.asarray(r, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    if r.shape != R.shape or r.ndim != 1:
        raise ValueError(
            f"r/R must be same-shape 1D, got {r.shape}, {R.shape}"
        )

    # 첫 음수 위치 찾기
    neg_idx = np.argmax(R < 0)
    if neg_idx == 0 and R[0] >= 0:
        # 모두 양수 → 전체 적분
        return float(np.trapezoid(R, r))

    # 음수가 처음 나타난 시점까지 적분
    return float(np.trapezoid(R[:neg_idx], r[:neg_idx]))


def integral_time_scale_from_R(
    tau: NDArray[np.float64],
    R: NDArray[np.float64],
) -> float:
    """적분 시간 스케일 T = ∫₀^∞ R(τ) dτ.

    Args:
        tau: 시간 랙 배열.
        R: 시간 자기상관.

    Returns:
        T.
    """
    return integral_length_scale_from_R(tau, R)


def taylor_microscale(
    r: NDArray[np.float64],
    R: NDArray[np.float64],
) -> float:
    """Taylor 미세 스케일 λ — R(r) ≈ 1 - (r/λ)² 이차 회귀.

    R = 1 - r²/λ² + O(r⁴)이므로 λ² = -2R(0)/R''(0).
    실제로는 처음 몇 점에서 이차 다항식 회귀.

    Args:
        r: 거리 배열.
        R: 상관 함수.

    Returns:
        λ.

    Raises:
        ValueError: 데이터 부족 또는 적합 불가.
    """
    r = np.asarray(r, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    if len(r) < 4:
        raise ValueError(f"need at least 4 points to fit, got {len(r)}")
    # 처음 R > 0.5 영역
    mask = R > 0.5
    if mask.sum() < 4:
        mask = np.zeros_like(R, dtype=bool)
        mask[: min(8, len(R))] = True

    rs = r[mask]
    Rs = R[mask]
    # f(r) = 1 - (r/λ)² → 회귀: f - 1 = -r²/λ² → 기울기 = -1/λ²
    coef = np.polyfit(rs ** 2, Rs - 1.0, 1)
    slope = coef[0]
    if slope >= -1e-30:
        raise ValueError(f"cannot fit Taylor microscale, slope={slope}")
    return float(np.sqrt(-1.0 / slope))


__all__ = [
    "spatial_autocorrelation",
    "temporal_autocorrelation",
    "integral_length_scale_from_R",
    "integral_time_scale_from_R",
    "taylor_microscale",
]
