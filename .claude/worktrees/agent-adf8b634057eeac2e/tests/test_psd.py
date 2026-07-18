"""Round 607 — Welch PSD + cross-spectrum + coherence."""

from __future__ import annotations

import numpy as np
import pytest


class TestWelchPSD:
    def test_basic_peak_detection(self) -> None:
        from naviertwin.core.flow_analysis.psd import welch_psd

        rng = np.random.default_rng(0)
        fs = 1000.0
        N = 4096
        t = np.arange(N) / fs
        # 50Hz 사인 + 노이즈
        x = np.sin(2 * np.pi * 50 * t) + 0.1 * rng.standard_normal(N)
        f, P = welch_psd(x, fs=fs, nperseg=512)
        # 피크가 50Hz 근처
        peak_f = f[P.argmax()]
        assert abs(peak_f - 50.0) < 5.0

    def test_returns_correct_shapes(self) -> None:
        from naviertwin.core.flow_analysis.psd import welch_psd

        rng = np.random.default_rng(1)
        x = rng.standard_normal(1024)
        f, P = welch_psd(x, fs=10.0, nperseg=128)
        assert f.shape == P.shape
        assert len(f) == 65  # 128//2 + 1
        assert np.all(P >= 0)

    def test_noverlap_default(self) -> None:
        from naviertwin.core.flow_analysis.psd import welch_psd

        rng = np.random.default_rng(2)
        x = rng.standard_normal(512)
        f, P = welch_psd(x, fs=1.0)
        assert len(f) > 0

    def test_hamming_window(self) -> None:
        from naviertwin.core.flow_analysis.psd import welch_psd

        rng = np.random.default_rng(3)
        x = rng.standard_normal(512)
        f, P = welch_psd(x, fs=1.0, window="hamming")
        assert len(f) > 0

    def test_boxcar_window(self) -> None:
        from naviertwin.core.flow_analysis.psd import welch_psd

        rng = np.random.default_rng(4)
        x = rng.standard_normal(256)
        f, P = welch_psd(x, fs=1.0, window="boxcar")
        assert np.all(P >= 0)

    def test_spectrum_scaling(self) -> None:
        from naviertwin.core.flow_analysis.psd import welch_psd

        rng = np.random.default_rng(5)
        x = rng.standard_normal(512)
        f, P = welch_psd(x, fs=1.0, scaling="spectrum")
        assert np.all(P >= 0)

    def test_detrend_none(self) -> None:
        from naviertwin.core.flow_analysis.psd import welch_psd

        x = np.ones(256) + np.linspace(0, 1, 256)
        f, P = welch_psd(x, fs=1.0, detrend="none")
        assert len(P) > 0

    def test_invalid_ndim_raises(self) -> None:
        from naviertwin.core.flow_analysis.psd import welch_psd

        with pytest.raises(ValueError, match="1D"):
            welch_psd(np.zeros((10, 5)))

    def test_too_short_raises(self) -> None:
        from naviertwin.core.flow_analysis.psd import welch_psd

        with pytest.raises(ValueError, match=">= 2"):
            welch_psd(np.array([1.0]))

    def test_invalid_window_raises(self) -> None:
        from naviertwin.core.flow_analysis.psd import welch_psd

        with pytest.raises(ValueError, match="window"):
            welch_psd(np.zeros(100), window="bogus")

    def test_invalid_noverlap_raises(self) -> None:
        from naviertwin.core.flow_analysis.psd import welch_psd

        with pytest.raises(ValueError, match="noverlap"):
            welch_psd(np.zeros(100), nperseg=64, noverlap=64)

    def test_invalid_scaling_raises(self) -> None:
        from naviertwin.core.flow_analysis.psd import welch_psd

        with pytest.raises(ValueError, match="scaling"):
            welch_psd(np.zeros(100), scaling="bogus")

    def test_invalid_nperseg_raises(self) -> None:
        from naviertwin.core.flow_analysis.psd import welch_psd

        with pytest.raises(ValueError, match="nperseg"):
            welch_psd(np.zeros(100), nperseg=0)


class TestCrossPSD:
    def test_basic(self) -> None:
        from naviertwin.core.flow_analysis.psd import cross_psd

        rng = np.random.default_rng(6)
        N = 1024
        x = rng.standard_normal(N)
        y = x + 0.1 * rng.standard_normal(N)
        f, Cxy = cross_psd(x, y, fs=100.0, nperseg=128)
        assert f.shape == Cxy.shape
        # complex array
        assert Cxy.dtype.kind == "c"

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.psd import cross_psd

        with pytest.raises(ValueError, match="same-shape"):
            cross_psd(np.zeros(100), np.zeros(80))


class TestCoherence:
    def test_identical_signals_unity(self) -> None:
        from naviertwin.core.flow_analysis.psd import coherence

        rng = np.random.default_rng(7)
        x = rng.standard_normal(2048)
        # 동일 신호 → 거의 1.0
        f, c = coherence(x, x, fs=10.0, nperseg=256)
        # 일부 주파수에서 0이 될 수 있지만 평균은 높음
        assert c.mean() > 0.5

    def test_independent_signals_low(self) -> None:
        from naviertwin.core.flow_analysis.psd import coherence

        rng1 = np.random.default_rng(8)
        rng2 = np.random.default_rng(9)
        x = rng1.standard_normal(4096)
        y = rng2.standard_normal(4096)
        f, c = coherence(x, y, fs=10.0, nperseg=256)
        # 독립 → 낮은 coherence
        assert c.mean() < 0.5

    def test_in_range_zero_one(self) -> None:
        from naviertwin.core.flow_analysis.psd import coherence

        rng = np.random.default_rng(10)
        x = rng.standard_normal(1024)
        y = rng.standard_normal(1024)
        f, c = coherence(x, y, fs=1.0, nperseg=128)
        assert np.all((c >= 0) & (c <= 1))
