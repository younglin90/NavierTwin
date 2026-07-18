"""Round 425 — Hilbert spectrum."""

from __future__ import annotations

import numpy as np


class TestHilbert:
    def test_constant_amp(self) -> None:
        from naviertwin.core.analysis.hilbert_spectrum import hilbert_amp_freq

        fs = 200
        t = np.linspace(0, 1, fs, endpoint=False)
        x = np.cos(2 * np.pi * 5 * t)
        amp, freq = hilbert_amp_freq(x, fs=fs)
        # amp ≈ 1
        assert np.allclose(amp[20:-20], 1.0, atol=0.05)
        # freq ≈ 5
        assert abs(freq[fs // 2] - 5.0) < 0.5
