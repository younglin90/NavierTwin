"""Round 596 — WaveletDiffusionNO fit/sample/error coverage."""

from __future__ import annotations

import numpy as np
import pytest


class TestWaveletDiffusionNOFull:
    def test_fit_and_sample(self) -> None:
        pytest.importorskip("pywt")
        from naviertwin.core.generative.wavelet_diffusion.wavelet_diffusion_no import (
            WaveletDiffusionNO,
        )

        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 32)).astype(np.float32)
        m = WaveletDiffusionNO(n_features=32, wavelet="db2", level=1,
                                n_steps=4, max_epochs=2)
        m.fit(X)
        assert m.is_fitted
        samples = m.sample(3, seed=1)
        assert samples.shape == (3, 32)
        assert np.isfinite(samples).all()

    def test_fit_wrong_shape_raises(self) -> None:
        pytest.importorskip("pywt")
        from naviertwin.core.generative.wavelet_diffusion.wavelet_diffusion_no import (
            WaveletDiffusionNO,
        )

        m = WaveletDiffusionNO(n_features=32)
        X_bad = np.zeros((5, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="n_features|shape"):
            m.fit(X_bad)

    def test_fit_1d_input_raises(self) -> None:
        pytest.importorskip("pywt")
        from naviertwin.core.generative.wavelet_diffusion.wavelet_diffusion_no import (
            WaveletDiffusionNO,
        )

        m = WaveletDiffusionNO(n_features=32)
        X_bad = np.zeros(32, dtype=np.float32)
        with pytest.raises(ValueError, match="2D"):
            m.fit(X_bad)

    def test_sample_before_fit_raises(self) -> None:
        pytest.importorskip("pywt")
        from naviertwin.core.generative.wavelet_diffusion.wavelet_diffusion_no import (
            WaveletDiffusionNO,
        )

        m = WaveletDiffusionNO(n_features=32)
        with pytest.raises(RuntimeError, match="fit"):
            m.sample(2)

    def test_pack_unpack_roundtrip(self) -> None:
        pytest.importorskip("pywt")
        from naviertwin.core.generative.wavelet_diffusion.wavelet_diffusion_no import (
            WaveletDiffusionNO,
        )

        rng = np.random.default_rng(7)
        X = rng.standard_normal((5, 64)).astype(np.float32)
        m = WaveletDiffusionNO(n_features=64, wavelet="haar", level=1)
        V = m._pack(X)
        assert V.ndim == 2
        X_rec = m._unpack(V)
        assert X_rec.shape == (5, 64)
        # round-trip should be close
        np.testing.assert_allclose(X_rec.astype(np.float32), X, atol=1e-4)
