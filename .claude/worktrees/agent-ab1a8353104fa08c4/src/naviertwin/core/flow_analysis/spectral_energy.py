"""유체 에너지 스펙트럼 분석 모듈.

속도 장(velocity field)에서 운동 에너지 스펙트럼을 계산하고
Kolmogorov -5/3 법칙 적합성을 평가한다.

References:
    Pope, S.B., "Turbulent Flows", Cambridge University Press, 2000.
    Kolmogorov, A.N., "The local structure of turbulence ...", 1941.

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> u = rng.standard_normal(256)
    >>> from naviertwin.core.flow_analysis.spectral_energy import (
    ...     energy_spectrum_1d, kolmogorov_slope
    ... )
    >>> k, E = energy_spectrum_1d(u)
    >>> slope, _ = kolmogorov_slope(k, E)
    >>> isinstance(slope, float)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import HAS_NATIVE_KERNELS, _kernels
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def energy_spectrum_1d(
    u: NDArray[np.float64],
    dx: float = 1.0,
    window: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """1D 속도 배열에서 파수-에너지 스펙트럼을 계산한다.

    FFT 기반으로 단면 에너지 E(k) = |û(k)|² / N 을 계산한다.
    양의 파수만 반환 (실수 신호 대칭성 활용).

    Args:
        u: (N,) 속도 배열.
        dx: 격자 간격 (공간 해상도).
        window: True이면 Hanning 창함수를 적용해 스펙트럴 누출 감소.

    Returns:
        Tuple[k, E]:
            k: (N//2,) 양의 파수 배열.
            E: (N//2,) 해당 파수의 에너지 스펙트럼.

    Raises:
        ValueError: u가 1D가 아닌 경우.
    """
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 1:
        raise ValueError(f"u must be 1D, got shape {u.shape}")

    N = len(u)
    if window:
        u = u * np.hanning(N)

    u_hat = np.fft.rfft(u)
    k = np.fft.rfftfreq(N, d=dx)  # 0부터 Nyquist까지

    # E(k) = |û|² / N²  (정규화)
    E = (np.abs(u_hat) ** 2) / (N * N)

    # 1-sided 스펙트럼: DC 제외 성분 × 2 (양측 대칭)
    E_onesided = E.copy()
    E_onesided[1:-1] *= 2.0

    logger.debug("에너지 스펙트럼 계산: N=%d, dx=%.4g", N, dx)
    return k, E_onesided


def energy_spectrum_2d(
    ux: NDArray[np.float64],
    uy: NDArray[np.float64],
    dx: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """2D 속도 장에서 방위각 평균 에너지 스펙트럼 E(k)를 계산한다.

    Args:
        ux: (Ny, Nx) x 방향 속도.
        uy: (Ny, Nx) y 방향 속도.
        dx: 격자 간격 (등방성 격자 가정).

    Returns:
        Tuple[k_radial, E_radial]:
            k_radial: 방사 파수 배열.
            E_radial: 방위각 평균 에너지.

    Raises:
        ValueError: ux, uy 형상이 일치하지 않는 경우.
    """
    ux = np.asarray(ux, dtype=np.float64)
    uy = np.asarray(uy, dtype=np.float64)
    if ux.shape != uy.shape or ux.ndim != 2:
        raise ValueError(
            f"ux and uy must be 2D with same shape, got {ux.shape} and {uy.shape}"
        )

    Ny, Nx = ux.shape
    ux_hat = np.fft.rfft2(ux)
    uy_hat = np.fft.rfft2(uy)
    E2d = (np.abs(ux_hat) ** 2 + np.abs(uy_hat) ** 2) / (Nx * Ny) ** 2

    # 파수 격자
    kx = np.fft.rfftfreq(Nx, d=dx)
    ky = np.fft.fftfreq(Ny, d=dx)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)

    # 방위각 평균: 방사 파수 bin 집계
    k_max = np.min([kx.max(), np.abs(ky).max()])
    n_bins = min(Nx // 2, Ny // 2)
    k_bins = np.linspace(0, k_max, n_bins + 1)
    k_radial = 0.5 * (k_bins[:-1] + k_bins[1:])
    native_radial = (
        getattr(_kernels, "radial_energy_sum", None)
        if HAS_NATIVE_KERNELS
        else None
    )
    if native_radial is not None:
        E_radial = np.asarray(native_radial(K, E2d, k_bins), dtype=np.float64)
    else:
        bin_idx = np.searchsorted(k_bins, K.ravel(), side="right") - 1
        valid = (bin_idx >= 0) & (bin_idx < n_bins)
        E_radial = np.bincount(
            bin_idx[valid],
            weights=E2d.ravel()[valid],
            minlength=n_bins,
        ).astype(np.float64)

    logger.debug("2D 에너지 스펙트럼: shape=%s, bins=%d", ux.shape, n_bins)
    return k_radial, E_radial


def kolmogorov_slope(
    k: NDArray[np.float64],
    E: NDArray[np.float64],
    k_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """에너지 스펙트럼의 로그-로그 기울기를 피팅해 Kolmogorov 법칙 적합성을 평가한다.

    관성 부분 영역(inertial subrange)에서 E(k) ∝ k^α 를 로그-로그 선형 회귀로 추정.
    이상적 난류: α ≈ -5/3 ≈ -1.667.

    Args:
        k: 파수 배열 (양의 값만).
        E: 에너지 스펙트럼 배열.
        k_range: (k_min, k_max) 피팅 범위. None이면 전체.

    Returns:
        Tuple[slope, r_squared]:
            slope: 로그-로그 기울기 (Kolmogorov: ≈ -5/3).
            r_squared: 결정계수 R².

    Raises:
        ValueError: 유효한 데이터 포인트가 2개 미만인 경우.
    """
    k = np.asarray(k, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)

    mask = (k > 0) & (E > 0)
    if k_range is not None:
        mask &= (k >= k_range[0]) & (k <= k_range[1])

    lk = np.log(k[mask])
    lE = np.log(E[mask])

    if len(lk) < 2:
        raise ValueError(
            "Not enough valid data points to fit "
            f"(need >= 2, got {len(lk)}). Check k_range or E values."
        )

    # 선형 회귀
    A = np.column_stack([lk, np.ones(len(lk))])
    result, _, _, _ = np.linalg.lstsq(A, lE, rcond=None)
    slope, intercept = result

    # R²
    y_pred = slope * lk + intercept
    ss_res = np.sum((lE - y_pred) ** 2)
    ss_tot = np.sum((lE - lE.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-30)

    logger.debug("Kolmogorov 기울기 피팅: slope=%.3f, R²=%.3f", slope, r2)
    return float(slope), float(r2)


def integral_length_scale(
    k: NDArray[np.float64],
    E: NDArray[np.float64],
) -> float:
    """에너지 스펙트럼에서 적분 길이 스케일 L을 계산한다.

    L = (π / 2u²) ∫ E(k)/k dk (von Kármán 정의)

    Args:
        k: 파수 배열 (k > 0).
        E: 에너지 스펙트럼.

    Returns:
        적분 길이 스케일 L.
    """
    k = np.asarray(k, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    mask = k > 0
    k, E = k[mask], E[mask]

    u2 = np.trapezoid(E, k)
    if u2 < 1e-30:
        return 0.0

    L = (np.pi / (2 * u2)) * np.trapezoid(E / k, k)
    return float(L)


__all__ = [
    "energy_spectrum_1d",
    "energy_spectrum_2d",
    "kolmogorov_slope",
    "integral_length_scale",
]
