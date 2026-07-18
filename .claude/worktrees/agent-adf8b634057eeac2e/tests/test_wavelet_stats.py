"""Round 577 — coverage uplift for flow_analysis.statistics.wavelet (was 0%)."""

from __future__ import annotations

import numpy as np
import pytest


class TestWavelet:
    def test_stft_fallback_shape(self) -> None:
        from naviertwin.core.flow_analysis.statistics.wavelet import stft_fallback

        rng = np.random.default_rng(0)
        sig = rng.standard_normal(256)
        out = stft_fallback(sig, dt=0.01, n_window=64, n_overlap=32)
        assert out["spectrogram"].shape[0] == 64 // 2 + 1
        assert out["frequencies"].shape == (64 // 2 + 1,)
        assert out["times"].shape == (out["spectrogram"].shape[1],)

    def test_stft_default_overlap(self) -> None:
        from naviertwin.core.flow_analysis.statistics.wavelet import stft_fallback

        # 256-sample, 4 cycles → ~16 samples/cycle → freq ≈ 1/(16 dt)
        n = 256
        sig = np.sin(np.linspace(0, 4 * 2 * np.pi, n, endpoint=False))
        out = stft_fallback(sig, dt=0.5, n_window=64)
        # power non-negative + finite
        assert (out["spectrogram"] >= 0).all()
        assert np.isfinite(out["spectrogram"]).all()

    def test_continuous_wavelet_or_skip(self) -> None:
        pywt = pytest.importorskip("pywt")
        from naviertwin.core.flow_analysis.statistics.wavelet import continuous_wavelet

        sig = np.sin(np.linspace(0, 8 * np.pi, 256))
        out = continuous_wavelet(sig, dt=0.01)
        assert out["coefficients"].shape[1] == 256
        assert out["frequencies"].shape == out["scales"].shape
        _ = pywt  # silence

    def test_continuous_wavelet_raises_without_pywt(self, monkeypatch) -> None:
        # simulate pywt missing
        import builtins

        real_import = builtins.__import__

        def block_pywt(name, *a, **kw):
            if name == "pywt":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block_pywt)
        from naviertwin.core.flow_analysis.statistics.wavelet import continuous_wavelet

        with pytest.raises(RuntimeError, match="PyWavelets"):
            continuous_wavelet(np.zeros(64))
