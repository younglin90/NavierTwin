"""신호 디노이징 — Savitzky-Golay, moving median, wavelet shrinkage.

CFD 측정 시계열에서 측정 노이즈/spike 제거. 후처리 전 신호 평활화.

상용 툴 대응:
    - MATLAB Signal Processing Toolbox: smoothdata, sgolayfilt, hampel
    - Tecplot 360: Smooth Data
    - SciPy: signal.savgol_filter, ndimage.median_filter

References:
    Savitzky, A. & Golay, M.J.E., "Smoothing and Differentiation of Data by
    Simplified Least Squares Procedures", Anal. Chem., 1964.

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = np.sin(np.linspace(0, 4*np.pi, 200)) + 0.3 * rng.standard_normal(200)
    >>> from naviertwin.core.flow_analysis.denoise import savgol_filter
    >>> y = savgol_filter(x, window_length=11, polyorder=3)
    >>> y.shape
    (200,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def savgol_filter(
    x: NDArray[np.float64],
    window_length: int,
    polyorder: int = 3,
    deriv: int = 0,
) -> NDArray[np.float64]:
    """Savitzky-Golay 다항 평활/미분 필터.

    Args:
        x: (N,) 신호.
        window_length: 윈도우 크기 (홀수).
        polyorder: 다항식 차수 (< window_length).
        deriv: 미분 차수 (0=평활, 1=1차 미분, ...).

    Returns:
        평활된 (또는 미분된) 신호. 양 끝은 거울 패딩.

    Raises:
        ValueError: 매개변수 오류.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if window_length < 3 or window_length % 2 == 0:
        raise ValueError(
            f"window_length must be odd and >= 3, got {window_length}"
        )
    if polyorder >= window_length:
        raise ValueError(
            f"polyorder ({polyorder}) must be < window_length ({window_length})"
        )
    if deriv < 0 or deriv > polyorder:
        raise ValueError(
            f"deriv must be in [0, polyorder={polyorder}], got {deriv}"
        )

    half = window_length // 2
    # 디자인 행렬 A_ij = j^i
    j = np.arange(-half, half + 1, dtype=np.float64)
    A = j[:, np.newaxis] ** np.arange(polyorder + 1, dtype=np.float64)
    # 의사역 → 가운데 행이 평활 계수
    pinv = np.linalg.pinv(A)
    # deriv차 도함수 계수: pinv[deriv] * deriv!
    from math import factorial

    coeffs = pinv[deriv] * factorial(deriv)

    padded = np.pad(x, half, mode="reflect")
    return np.convolve(padded, coeffs[::-1], mode="valid")


def moving_average(
    x: NDArray[np.float64],
    window_length: int,
) -> NDArray[np.float64]:
    """단순 이동 평균 (uniform kernel).

    Args:
        x: 1D 신호.
        window_length: 홀수 윈도우 크기.

    Returns:
        평균된 신호.

    Raises:
        ValueError: window_length 오류.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if window_length < 1:
        raise ValueError(f"window_length must be >= 1, got {window_length}")
    if window_length % 2 == 0:
        raise ValueError(f"window_length must be odd, got {window_length}")

    half = window_length // 2
    padded = np.pad(x, half, mode="reflect")
    kernel = np.ones(window_length) / window_length
    return np.convolve(padded, kernel, mode="valid")


def moving_median(
    x: NDArray[np.float64],
    window_length: int,
) -> NDArray[np.float64]:
    """이동 중앙값 — 임펄스 노이즈/스파이크 제거에 강함.

    Args:
        x: 1D 신호.
        window_length: 홀수.

    Returns:
        평활 신호.

    Raises:
        ValueError: 매개변수 오류.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if window_length < 1 or window_length % 2 == 0:
        raise ValueError(
            f"window_length must be odd and >= 1, got {window_length}"
        )

    n = len(x)
    half = window_length // 2
    padded = np.pad(x, half, mode="reflect")
    wins = np.lib.stride_tricks.sliding_window_view(padded, window_length)[:n]
    return np.median(wins, axis=1)


def hampel_filter(
    x: NDArray[np.float64],
    window_length: int = 7,
    n_sigmas: float = 3.0,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Hampel 필터 — 윈도우 중앙값 ± n·MAD를 벗어나는 값을 중앙값으로 대체.

    이동 중앙값 디노이징의 진화 버전 (스파이크만 골라 제거).

    Args:
        x: 1D 신호.
        window_length: 홀수 윈도우.
        n_sigmas: 임계값 (MAD의 배수).

    Returns:
        (cleaned, outlier_mask): 정리된 신호 + 이상치 마스크.

    Raises:
        ValueError: 매개변수 오류.
    """
    x = np.asarray(x, dtype=np.float64).ravel().copy()
    if window_length < 3 or window_length % 2 == 0:
        raise ValueError(
            f"window_length must be odd and >= 3, got {window_length}"
        )
    if n_sigmas <= 0:
        raise ValueError(f"n_sigmas must be > 0, got {n_sigmas}")

    n = len(x)
    half = window_length // 2
    padded = np.pad(x, half, mode="reflect")
    cleaned = x.copy()
    wins = np.lib.stride_tricks.sliding_window_view(padded, window_length)[:n]
    med = np.median(wins, axis=1)
    mad = np.median(np.abs(wins - med[:, np.newaxis]), axis=1) * 1.4826
    mask = (mad >= 1e-30) & (np.abs(x - med) > n_sigmas * mad)
    cleaned[mask] = med[mask]
    return cleaned, mask


def soft_threshold(
    coeffs: NDArray[np.float64], threshold: float
) -> NDArray[np.float64]:
    """소프트 임계값 wavelet shrinkage (Donoho).

    Args:
        coeffs: 계수.
        threshold: 임계값.

    Returns:
        sign(c) * max(|c| - λ, 0).
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0.0)


def hard_threshold(
    coeffs: NDArray[np.float64], threshold: float
) -> NDArray[np.float64]:
    """하드 임계값."""
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    return np.where(np.abs(coeffs) > threshold, coeffs, 0.0)


def universal_threshold(
    sigma: float, n: int
) -> float:
    """Donoho-Johnstone universal threshold λ = σ √(2 ln N).

    Args:
        sigma: 노이즈 표준편차 추정값.
        n: 신호 길이.

    Returns:
        임계값.
    """
    if sigma <= 0 or n < 1:
        raise ValueError(f"sigma > 0 and n >= 1 required, got {sigma}, {n}")
    return sigma * float(np.sqrt(2.0 * np.log(n)))


def estimate_noise_sigma_mad(
    x: NDArray[np.float64],
) -> float:
    """차분 MAD 기반 노이즈 σ 추정 — wavelet 디노이징 표준.

    σ ≈ MAD(diff(x)) / (1.4826 · √2)

    Args:
        x: 1D 신호.

    Returns:
        σ 추정값.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) < 2:
        return 0.0
    d = np.diff(x)
    mad = float(np.median(np.abs(d - np.median(d))))
    return mad * 1.4826 / float(np.sqrt(2.0))


__all__ = [
    "savgol_filter",
    "moving_average",
    "moving_median",
    "hampel_filter",
    "soft_threshold",
    "hard_threshold",
    "universal_threshold",
    "estimate_noise_sigma_mad",
]
