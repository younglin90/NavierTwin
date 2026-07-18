"""Synchrosqueezing — Daubechies et al. 2011, 시간-주파수 재할당.

간단 STFT-기반.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.synchrosqueeze import synchrosqueeze_stft
    >>> t = np.linspace(0, 1, 256, endpoint=False)
    >>> x = np.cos(2*np.pi*10*t)
    >>> Tx, freqs = synchrosqueeze_stft(x, fs=256, win=64)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _stft(x, win, hop):
    n = len(x)
    n_seg = (n - win) // hop + 1
    w = np.hanning(win)
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop][:n_seg]
    return np.fft.fft(frames * w, axis=1).T


def synchrosqueeze_stft(
    x: NDArray[np.float64], *, fs: float = 1.0, win: int = 64, hop: int = 16,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """간단 synchrosqueeze: STFT magnitude → reassign by argmax frequency."""
    X = _stft(np.asarray(x, dtype=np.float64), win, hop)
    freqs = np.fft.fftfreq(win, d=1.0 / fs)
    n_seg = X.shape[1]
    Tx = np.zeros_like(X)
    idx = np.argmax(np.abs(X), axis=0)
    cols = np.arange(n_seg)
    Tx[idx, cols] = X[idx, cols]
    return Tx, freqs


__all__ = ["synchrosqueeze_stft"]
