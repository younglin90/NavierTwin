"""시계열 신호의 FFT 파워 스펙트럼 + dominant 주파수 검출.

CFD 시계열(점 probe, 모드 계수 등)에서 주기 성분을 파악.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.spectrum import power_spectrum
    >>> t = np.linspace(0, 1, 1000, endpoint=False)
    >>> x = np.sin(2*np.pi*5*t) + 0.1*np.sin(2*np.pi*50*t)
    >>> f, P = power_spectrum(x, dt=1/1000)
    >>> abs(f[np.argmax(P)] - 5.0) < 0.5
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def power_spectrum(
    x: NDArray[np.float64],
    dt: float = 1.0,
    *,
    detrend: bool = True,
    window: str = "hann",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """신호의 one-sided 파워 스펙트럼 (Hz 단위, dt=샘플 간격).

    Returns:
        (frequencies, power).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        raise ValueError("n >= 2")
    if detrend:
        x = x - x.mean()
    if window == "hann":
        w = np.hanning(n)
    elif window == "hamming":
        w = np.hamming(n)
    elif window == "none":
        w = np.ones(n)
    else:
        raise ValueError(f"unknown window: {window}")
    X = np.fft.rfft(x * w)
    freqs = np.fft.rfftfreq(n, d=dt)
    # normalize by window energy
    wn = np.sum(w ** 2)
    power = (np.abs(X) ** 2) / (wn + 1e-30)
    return freqs, power


def dominant_frequencies(
    x: NDArray[np.float64],
    dt: float = 1.0,
    *,
    top_k: int = 3,
) -> list[tuple[float, float]]:
    """상위 top_k 개 (freq, power) 튜플 (DC 제외)."""
    f, P = power_spectrum(x, dt=dt)
    # DC 제외
    f, P = f[1:], P[1:]
    k = int(min(top_k, f.size))
    return _kernels.dominant_frequencies_from_power(f, P, k)


__all__ = ["power_spectrum", "dominant_frequencies"]
