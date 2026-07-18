"""Round 619 — signal denoising (Savgol, moving median, Hampel, wavelet shrinkage)."""

from __future__ import annotations

import numpy as np
import pytest


class TestSavgol:
    def test_smooth_sine(self) -> None:
        from naviertwin.core.flow_analysis.denoise import savgol_filter

        rng = np.random.default_rng(0)
        x = np.sin(np.linspace(0, 4 * np.pi, 200))
        noisy = x + 0.3 * rng.standard_normal(200)
        y = savgol_filter(noisy, window_length=21, polyorder=3)
        # MSE 감소
        mse_before = np.mean((noisy - x) ** 2)
        mse_after = np.mean((y - x) ** 2)
        assert mse_after < mse_before

    def test_first_derivative(self) -> None:
        from naviertwin.core.flow_analysis.denoise import savgol_filter

        # f(x) = x² → f'(x) = 2x; SG returns df/d(index)
        t = np.linspace(0, 1, 100)
        h = t[1] - t[0]
        f = t ** 2
        df = savgol_filter(f, window_length=11, polyorder=3, deriv=1)
        # 중앙 영역만 검증 (양 끝 boundary는 reflect padding 영향)
        df_phys = df / h
        np.testing.assert_allclose(df_phys[20:-20], 2 * t[20:-20], atol=0.05)

    def test_invalid_window_even(self) -> None:
        from naviertwin.core.flow_analysis.denoise import savgol_filter

        with pytest.raises(ValueError, match="window_length"):
            savgol_filter(np.zeros(100), window_length=10)

    def test_invalid_polyorder(self) -> None:
        from naviertwin.core.flow_analysis.denoise import savgol_filter

        with pytest.raises(ValueError, match="polyorder"):
            savgol_filter(np.zeros(100), window_length=5, polyorder=5)

    def test_invalid_deriv(self) -> None:
        from naviertwin.core.flow_analysis.denoise import savgol_filter

        with pytest.raises(ValueError, match="deriv"):
            savgol_filter(np.zeros(100), window_length=11, polyorder=3, deriv=4)


class TestMovingAverage:
    def test_constant(self) -> None:
        from naviertwin.core.flow_analysis.denoise import moving_average

        x = np.ones(50) * 5.0
        y = moving_average(x, window_length=5)
        np.testing.assert_allclose(y, 5.0)

    def test_linear(self) -> None:
        from naviertwin.core.flow_analysis.denoise import moving_average

        x = np.linspace(0, 1, 100)
        y = moving_average(x, window_length=11)
        # 선형 입력 → 선형 출력 (양 끝 제외)
        np.testing.assert_allclose(y[10:-10], x[10:-10], atol=1e-10)

    def test_invalid_window_even_raises(self) -> None:
        from naviertwin.core.flow_analysis.denoise import moving_average

        with pytest.raises(ValueError, match="odd"):
            moving_average(np.zeros(50), window_length=4)

    def test_invalid_window_zero_raises(self) -> None:
        from naviertwin.core.flow_analysis.denoise import moving_average

        with pytest.raises(ValueError, match=">= 1"):
            moving_average(np.zeros(50), window_length=0)


class TestMovingMedian:
    def test_spike_removal(self) -> None:
        from naviertwin.core.flow_analysis.denoise import moving_median

        # 평탄 신호 + 강한 spike
        x = np.ones(50)
        x[25] = 100.0
        y = moving_median(x, window_length=5)
        # spike가 사라짐
        assert abs(y[25] - 1.0) < 1e-10

    def test_invalid_window_raises(self) -> None:
        from naviertwin.core.flow_analysis.denoise import moving_median

        with pytest.raises(ValueError, match="odd"):
            moving_median(np.zeros(50), window_length=4)


class TestHampel:
    def test_outliers_replaced(self) -> None:
        from naviertwin.core.flow_analysis.denoise import hampel_filter

        rng = np.random.default_rng(1)
        x = rng.standard_normal(200)
        x[50] = 100.0
        x[120] = -50.0
        cleaned, mask = hampel_filter(x, window_length=11, n_sigmas=3.0)
        # spike 위치가 마스크됨
        assert mask[50] or mask[120]
        # 정리된 신호의 값이 원본보다 작음
        assert abs(cleaned[50]) < 50

    def test_no_outliers_no_change(self) -> None:
        from naviertwin.core.flow_analysis.denoise import hampel_filter

        rng = np.random.default_rng(2)
        x = rng.standard_normal(100)
        cleaned, mask = hampel_filter(x, window_length=7, n_sigmas=10.0)
        # 임계값 매우 큼 → 변화 거의 없음
        np.testing.assert_allclose(cleaned, x, atol=1e-10)
        assert not mask.any()

    def test_invalid_window_raises(self) -> None:
        from naviertwin.core.flow_analysis.denoise import hampel_filter

        with pytest.raises(ValueError, match="window_length"):
            hampel_filter(np.zeros(100), window_length=4)

    def test_invalid_sigma_raises(self) -> None:
        from naviertwin.core.flow_analysis.denoise import hampel_filter

        with pytest.raises(ValueError, match="n_sigmas"):
            hampel_filter(np.zeros(100), n_sigmas=0)


class TestThresholding:
    def test_soft_threshold(self) -> None:
        from naviertwin.core.flow_analysis.denoise import soft_threshold

        c = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
        y = soft_threshold(c, threshold=1.0)
        # |c| < λ → 0; |c| >= λ → sign * (|c| - λ)
        np.testing.assert_allclose(y, [-2.0, 0.0, 0.0, 0.0, 2.0])

    def test_hard_threshold(self) -> None:
        from naviertwin.core.flow_analysis.denoise import hard_threshold

        c = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
        y = hard_threshold(c, threshold=1.0)
        np.testing.assert_allclose(y, [-3.0, 0.0, 0.0, 0.0, 3.0])

    def test_negative_threshold_soft_raises(self) -> None:
        from naviertwin.core.flow_analysis.denoise import soft_threshold

        with pytest.raises(ValueError, match="threshold"):
            soft_threshold(np.zeros(5), threshold=-1.0)

    def test_negative_threshold_hard_raises(self) -> None:
        from naviertwin.core.flow_analysis.denoise import hard_threshold

        with pytest.raises(ValueError, match="threshold"):
            hard_threshold(np.zeros(5), threshold=-0.1)


class TestUniversalThreshold:
    def test_basic(self) -> None:
        from naviertwin.core.flow_analysis.denoise import universal_threshold

        # σ=1, N=1024 → λ = √(2 ln 1024) ≈ 3.72
        lam = universal_threshold(sigma=1.0, n=1024)
        assert abs(lam - np.sqrt(2 * np.log(1024))) < 1e-10

    def test_invalid_sigma(self) -> None:
        from naviertwin.core.flow_analysis.denoise import universal_threshold

        with pytest.raises(ValueError, match="sigma"):
            universal_threshold(sigma=0.0, n=100)


class TestNoiseEstimate:
    def test_unit_normal(self) -> None:
        from naviertwin.core.flow_analysis.denoise import estimate_noise_sigma_mad

        rng = np.random.default_rng(3)
        x = rng.standard_normal(2000)
        sigma = estimate_noise_sigma_mad(x)
        # σ ≈ 1
        assert 0.7 < sigma < 1.3

    def test_short_returns_zero(self) -> None:
        from naviertwin.core.flow_analysis.denoise import estimate_noise_sigma_mad

        sigma = estimate_noise_sigma_mad(np.array([1.0]))
        assert sigma == 0.0
