"""Short-Time Fourier Transform — 시간-주파수 스펙트로그램.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.stft import stft
    >>> fs = 1000
    >>> t = np.arange(0, 1, 1/fs)
    >>> x = np.sin(2*np.pi*50*t)
    >>> f, T, Z = stft(x, fs, window=128, overlap=64)
    >>> Z.shape[0] == 65
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def stft(
    x: NDArray[np.float64], fs: float = 1.0,
    *, window: int = 256, overlap: int = 128, win_type: str = "hann",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray]:
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    step = window - overlap
    if step <= 0:
        raise ValueError("overlap < window 필요")
    n_frames = 1 + max(0, (n - window) // step)
    if win_type == "hann":
        w = np.hanning(window)
    elif win_type == "hamming":
        w = np.hamming(window)
    else:
        w = np.ones(window)
    frames = np.lib.stride_tricks.sliding_window_view(x, window)[::step][:n_frames]
    Z = np.fft.rfft(frames * w, axis=1).T
    freqs = np.fft.rfftfreq(window, d=1.0 / fs)
    times = (np.arange(n_frames) * step + window / 2) / fs
    return freqs, times, Z


def spectrogram(Z: NDArray) -> NDArray[np.float64]:
    return np.abs(Z) ** 2


__all__ = ["stft", "spectrogram"]
