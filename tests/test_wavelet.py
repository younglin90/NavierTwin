"""Round 184 — Haar wavelet."""

from __future__ import annotations

import numpy as np


class TestHaar:
    def test_roundtrip(self) -> None:
        from naviertwin.core.analysis.wavelet import haar_forward, haar_inverse

        x = np.arange(16, dtype=float)
        c = haar_forward(x, level=3)
        y = haar_inverse(c)
        assert np.allclose(x, y, atol=1e-12)

    def test_threshold_denoise(self) -> None:
        from naviertwin.core.analysis.wavelet import (
            haar_forward,
            haar_inverse,
            haar_threshold,
        )

        rng = np.random.default_rng(0)
        n = 256
        t = np.linspace(0, 1, n)
        clean = np.sin(2 * np.pi * 3 * t)
        noisy = clean + 0.2 * rng.standard_normal(n)
        c = haar_forward(noisy, level=4)
        c_t = haar_threshold(c, tau=0.1)
        denoised = haar_inverse(c_t)
        # 노이즈 감소
        assert np.linalg.norm(denoised - clean) < np.linalg.norm(noisy - clean)

    def test_invalid_length(self) -> None:
        import pytest as pt

        from naviertwin.core.analysis.wavelet import haar_forward

        with pt.raises(ValueError):
            haar_forward(np.arange(7, dtype=float))
