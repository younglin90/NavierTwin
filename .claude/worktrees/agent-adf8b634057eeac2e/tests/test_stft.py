"""Round 241 — STFT."""

from __future__ import annotations

import numpy as np


class TestSTFT:
    def test_tone_peak(self) -> None:
        from naviertwin.core.analysis.stft import spectrogram, stft

        fs = 2000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 100 * t)
        f, T, Z = stft(x, fs, window=256, overlap=128)
        S = spectrogram(Z)
        # 가장 강한 frequency bin ≈ 100 Hz
        peak_freq = f[np.argmax(S.mean(axis=1))]
        assert abs(peak_freq - 100.0) < 10

    def test_shape(self) -> None:
        from naviertwin.core.analysis.stft import stft

        x = np.zeros(1000)
        f, T, Z = stft(x, fs=500, window=128, overlap=64)
        assert Z.shape[0] == 128 // 2 + 1
        assert Z.shape[1] > 0

    def test_invalid(self) -> None:
        import pytest

        from naviertwin.core.analysis.stft import stft

        with pytest.raises(ValueError):
            stft(np.zeros(100), window=128, overlap=128)
