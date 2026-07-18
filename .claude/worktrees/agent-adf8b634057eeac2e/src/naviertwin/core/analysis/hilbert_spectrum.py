"""Hilbert spectrum — instantaneous amplitude/frequency via FFT-based Hilbert.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.hilbert_spectrum import hilbert_amp_freq
    >>> t = np.linspace(0, 1, 200, endpoint=False)
    >>> x = np.cos(2*np.pi*5*t)
    >>> amp, freq = hilbert_amp_freq(x, fs=200)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def hilbert_transform(x: NDArray[np.float64]) -> NDArray[np.complex128]:
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    return np.fft.ifft(X * h)


def hilbert_amp_freq(
    x: NDArray[np.float64], *, fs: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    z = hilbert_transform(np.asarray(x))
    amp = np.abs(z)
    phase = np.unwrap(np.angle(z))
    freq = np.diff(phase) / (2 * np.pi) * fs
    freq = np.concatenate([[freq[0]], freq])
    return amp, freq


__all__ = ["hilbert_amp_freq", "hilbert_transform"]
