"""Spectral 연산 — 주기 도메인에서 FFT 미분 + 저역/고역 필터.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.spectral import spectral_derivative_1d
    >>> x = np.linspace(0, 2*np.pi, 64, endpoint=False)
    >>> df = spectral_derivative_1d(np.sin(x), L=2*np.pi)
    >>> np.max(np.abs(df - np.cos(x))) < 1e-8
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def spectral_derivative_1d(
    f: NDArray[np.float64], L: float, order: int = 1,
) -> NDArray[np.float64]:
    """주기 신호의 n차 스펙트럴 미분."""
    n = f.size
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    F = np.fft.fft(f)
    dF = ((1j * k) ** order) * F
    return np.real(np.fft.ifft(dF))


def spectral_derivative_2d(
    f: NDArray[np.float64], Lx: float, Ly: float,
    *, order: tuple[int, int] = (1, 0),
) -> NDArray[np.float64]:
    """2D 스펙트럴 미분. f shape: (ny, nx). order=(ix, iy)."""
    ny, nx = f.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    F = np.fft.fft2(f)
    ix, iy = order
    dF = ((1j * KX) ** ix) * ((1j * KY) ** iy) * F
    return np.real(np.fft.ifft2(dF))


def lowpass_filter_1d(
    f: NDArray[np.float64], L: float, cutoff: float,
) -> NDArray[np.float64]:
    """|k| > cutoff 성분 제거."""
    n = f.size
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    F = np.fft.fft(f)
    F[np.abs(k) > cutoff] = 0.0
    return np.real(np.fft.ifft(F))


def highpass_filter_1d(
    f: NDArray[np.float64], L: float, cutoff: float,
) -> NDArray[np.float64]:
    n = f.size
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    F = np.fft.fft(f)
    F[np.abs(k) < cutoff] = 0.0
    return np.real(np.fft.ifft(F))


__all__ = [
    "spectral_derivative_1d",
    "spectral_derivative_2d",
    "lowpass_filter_1d",
    "highpass_filter_1d",
]
