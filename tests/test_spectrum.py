"""Round 88 — FFT power spectrum."""

from __future__ import annotations

import numpy as np
import pytest


class TestSpectrum:
    def test_single_tone_peak(self) -> None:
        from naviertwin.core.analysis.spectrum import power_spectrum

        fs = 1000.0
        t = np.arange(0, 2.0, 1 / fs)
        x = np.sin(2 * np.pi * 13.0 * t)
        f, P = power_spectrum(x, dt=1 / fs)
        peak = f[np.argmax(P)]
        assert abs(peak - 13.0) < 0.5

    def test_dominant(self) -> None:
        from naviertwin.core.analysis.spectrum import dominant_frequencies

        fs = 1000.0
        t = np.arange(0, 2.0, 1 / fs)
        x = np.sin(2 * np.pi * 7 * t) + 0.5 * np.sin(2 * np.pi * 23 * t)
        doms = dominant_frequencies(x, dt=1 / fs, top_k=2)
        freqs = sorted(f for f, _ in doms)
        # Hann leakage 로 ±1 Hz 정도 이동 가능
        assert abs(freqs[0] - 7.0) < 1.0
        assert abs(freqs[1] - 23.0) < 1.0

    def test_invalid(self) -> None:
        from naviertwin.core.analysis.spectrum import power_spectrum

        with pytest.raises(ValueError):
            power_spectrum(np.array([1.0]))
        with pytest.raises(ValueError):
            power_spectrum(np.zeros(10), window="bogus")

    def test_window_options(self) -> None:
        from naviertwin.core.analysis.spectrum import power_spectrum

        x = np.random.default_rng(0).standard_normal(256)
        for w in ("hann", "hamming", "none"):
            f, P = power_spectrum(x, window=w)
            assert len(f) == len(P)
