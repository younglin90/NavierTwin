"""Round 426 — synchrosqueeze."""

from __future__ import annotations

import numpy as np


class TestSync:
    def test_concentrates_on_freq(self) -> None:
        from naviertwin.core.analysis.synchrosqueeze import synchrosqueeze_stft

        fs = 256
        t = np.linspace(0, 1, fs, endpoint=False)
        x = np.cos(2 * np.pi * 30 * t)
        Tx, freqs = synchrosqueeze_stft(x, fs=fs, win=64, hop=16)
        # mostly zeros, peak at f≈30
        non_zero_per_seg = (np.abs(Tx) > 1e-6).sum(axis=0)
        assert (non_zero_per_seg <= 2).all()
