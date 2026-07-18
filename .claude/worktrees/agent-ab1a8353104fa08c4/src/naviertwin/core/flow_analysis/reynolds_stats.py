"""Reynolds 분해 및 난류 통계 후처리.

CFD 시간 시계열에서 평균(mean), 변동(fluctuation), Reynolds 응력 텐서,
난류 운동 에너지(TKE), 왜도/첨도, 시간평균 RMS 등 통계량을 추출한다.

상용 툴 대응:
    - Tecplot 360: Time-Average / Fluctuation
    - Ansys CFD-Post: Statistics → mean / RMS / fluctuation
    - EnSight: Time history → RMS

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> u = 1.0 + 0.1 * rng.standard_normal((100, 30))  # (n_t, n_space)
    >>> from naviertwin.core.flow_analysis.reynolds_stats import (
    ...     mean_field, fluctuation, rms, reynolds_stress_2d
    ... )
    >>> u_mean = mean_field(u)
    >>> u_mean.shape
    (30,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def mean_field(u: NDArray[np.float64], axis: int = 0) -> NDArray[np.float64]:
    """시간 또는 앙상블 평균.

    Args:
        u: (n_samples, ...) 데이터. axis 따라 평균.
        axis: 평균 차원 (기본 0 = 시간).

    Returns:
        평균 장.
    """
    u = np.asarray(u, dtype=np.float64)
    return u.mean(axis=axis)


def fluctuation(
    u: NDArray[np.float64], axis: int = 0
) -> NDArray[np.float64]:
    """변동 성분 u' = u - <u>.

    Reynolds 분해의 변동 부분.

    Args:
        u: 데이터.
        axis: 평균이 취해지는 축.

    Returns:
        u'(t, x) — u와 같은 형상.
    """
    u = np.asarray(u, dtype=np.float64)
    return u - u.mean(axis=axis, keepdims=True)


def rms(u: NDArray[np.float64], axis: int = 0) -> NDArray[np.float64]:
    """변동의 RMS = √<u'²>.

    Args:
        u: 데이터.
        axis: 시간 축.

    Returns:
        RMS 장.
    """
    up = fluctuation(u, axis=axis)
    return np.sqrt(np.mean(up * up, axis=axis))


def reynolds_stress_2d(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    axis: int = 0,
) -> dict[str, NDArray[np.float64]]:
    """2D Reynolds 응력 텐서 성분 <u'u'>, <v'v'>, <u'v'>.

    Args:
        u: x 방향 속도 시계열.
        v: y 방향 속도 시계열.
        axis: 시간 축.

    Returns:
        {"uu": ..., "vv": ..., "uv": ...} dict.

    Raises:
        ValueError: u와 v 형상 불일치.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError(f"u and v shapes mismatch: {u.shape} vs {v.shape}")

    up = fluctuation(u, axis=axis)
    vp = fluctuation(v, axis=axis)

    return {
        "uu": np.mean(up * up, axis=axis),
        "vv": np.mean(vp * vp, axis=axis),
        "uv": np.mean(up * vp, axis=axis),
    }


def reynolds_stress_3d(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    w: NDArray[np.float64],
    axis: int = 0,
) -> dict[str, NDArray[np.float64]]:
    """3D Reynolds 응력 텐서 6 성분.

    Args:
        u, v, w: 3D 속도 시계열.
        axis: 시간 축.

    Returns:
        {"uu", "vv", "ww", "uv", "uw", "vw"} dict.

    Raises:
        ValueError: 형상 불일치.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if not (u.shape == v.shape == w.shape):
        raise ValueError(
            f"u/v/w shapes mismatch: {u.shape}, {v.shape}, {w.shape}"
        )

    up = fluctuation(u, axis=axis)
    vp = fluctuation(v, axis=axis)
    wp = fluctuation(w, axis=axis)

    return {
        "uu": np.mean(up * up, axis=axis),
        "vv": np.mean(vp * vp, axis=axis),
        "ww": np.mean(wp * wp, axis=axis),
        "uv": np.mean(up * vp, axis=axis),
        "uw": np.mean(up * wp, axis=axis),
        "vw": np.mean(vp * wp, axis=axis),
    }


def turbulent_kinetic_energy(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    w: NDArray[np.float64] | None = None,
    axis: int = 0,
) -> NDArray[np.float64]:
    """난류 운동 에너지 k = ½ <u_i' u_i'>.

    Args:
        u, v: 평면 또는 3D 속도 시계열.
        w: 선택적 z 속도 시계열. 없으면 2D 가정.
        axis: 시간 축.

    Returns:
        TKE 장.
    """
    up = fluctuation(u, axis=axis)
    vp = fluctuation(v, axis=axis)
    tke = 0.5 * (np.mean(up * up, axis=axis) + np.mean(vp * vp, axis=axis))
    if w is not None:
        wp = fluctuation(w, axis=axis)
        tke = tke + 0.5 * np.mean(wp * wp, axis=axis)
    return tke


def turbulence_intensity(
    u: NDArray[np.float64],
    v: NDArray[np.float64] | None = None,
    w: NDArray[np.float64] | None = None,
    axis: int = 0,
) -> NDArray[np.float64]:
    """난류 강도 I = √(2k/3) / U_ref, 여기서 U_ref = |<u>|.

    1D인 경우: I = u_rms / |<u>|.

    Args:
        u: 주방향 속도 시계열.
        v, w: 추가 성분 (선택).
        axis: 시간 축.

    Returns:
        난류 강도 (무차원).
    """
    if v is None:
        u_rms = rms(u, axis=axis)
        u_mean = np.abs(mean_field(u, axis=axis)) + 1e-30
        return u_rms / u_mean

    k = turbulent_kinetic_energy(u, v, w, axis=axis)
    u_mean = mean_field(u, axis=axis)
    if v is not None:
        v_mean = mean_field(v, axis=axis)
        u_ref = np.sqrt(u_mean ** 2 + v_mean ** 2)
        if w is not None:
            w_mean = mean_field(w, axis=axis)
            u_ref = np.sqrt(u_ref ** 2 + w_mean ** 2)
    else:
        u_ref = np.abs(u_mean)
    u_ref = np.maximum(u_ref, 1e-30)
    return np.sqrt(2.0 * k / 3.0) / u_ref


def skewness(u: NDArray[np.float64], axis: int = 0) -> NDArray[np.float64]:
    """변동의 왜도 S = <u'³>/<u'²>^(3/2).

    Args:
        u: 시계열.
        axis: 시간 축.

    Returns:
        왜도 (정규분포는 0).
    """
    up = fluctuation(u, axis=axis)
    var = np.mean(up * up, axis=axis)
    third = np.mean(up ** 3, axis=axis)
    return third / np.maximum(var ** 1.5, 1e-30)


def flatness(u: NDArray[np.float64], axis: int = 0) -> NDArray[np.float64]:
    """변동의 첨도 F = <u'⁴>/<u'²>² (정규분포는 3).

    Args:
        u: 시계열.
        axis: 시간 축.

    Returns:
        첨도.
    """
    up = fluctuation(u, axis=axis)
    var = np.mean(up * up, axis=axis)
    fourth = np.mean(up ** 4, axis=axis)
    return fourth / np.maximum(var * var, 1e-30)


def running_statistics(
    u: NDArray[np.float64], axis: int = 0,
) -> dict[str, NDArray[np.float64]]:
    """누적 평균과 누적 RMS (실시간 모니터링용).

    Welford 알고리즘 기반.

    Args:
        u: (n_samples, ...) 시계열.
        axis: 시간 축 (현재는 0만 지원).

    Returns:
        {"mean": (n_samples, ...), "rms": (n_samples, ...)} —
        시점 t까지의 누적 통계.

    Raises:
        ValueError: axis가 0이 아닌 경우 (현재 미지원).
    """
    if axis != 0:
        raise ValueError(f"running_statistics currently supports axis=0 only, got {axis}")
    u = np.asarray(u, dtype=np.float64)
    n = u.shape[0]
    if n == 0:
        return {"mean": np.zeros_like(u), "rms": np.zeros_like(u)}

    counts = np.arange(1, n + 1, dtype=np.float64).reshape((n,) + (1,) * (u.ndim - 1))
    csum = np.cumsum(u, axis=0)
    mean = csum / counts
    csum2 = np.cumsum(u * u, axis=0)
    var = np.maximum(csum2 / counts - mean * mean, 0.0)
    rms_arr = np.sqrt(var)
    rms_arr[0] = 0.0
    return {"mean": mean, "rms": rms_arr}


__all__ = [
    "mean_field",
    "fluctuation",
    "rms",
    "reynolds_stress_2d",
    "reynolds_stress_3d",
    "turbulent_kinetic_energy",
    "turbulence_intensity",
    "skewness",
    "flatness",
    "running_statistics",
]
