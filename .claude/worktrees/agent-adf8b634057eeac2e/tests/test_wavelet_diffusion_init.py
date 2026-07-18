"""Round 588 — WaveletDiffusionNO init/error coverage."""

from __future__ import annotations

import builtins

import pytest


class TestWaveletDiffusionNO:
    def test_init_defaults(self) -> None:
        from naviertwin.core.generative.wavelet_diffusion.wavelet_diffusion_no import (
            WaveletDiffusionNO,
        )

        m = WaveletDiffusionNO(n_features=32)
        assert m.n_features == 32
        assert m.wavelet == "db2"
        assert m.level == 1
        assert m.is_fitted is False

    def test_pack_requires_pywt(self, monkeypatch) -> None:
        from naviertwin.core.generative.wavelet_diffusion import wavelet_diffusion_no

        real_import = builtins.__import__

        def block(name, *a, **kw):
            if name == "pywt":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block)
        with pytest.raises(RuntimeError, match="pywt"):
            wavelet_diffusion_no._require_pywt()
