"""Round 610 — two-point spatial/temporal autocorrelation + scales."""

from __future__ import annotations

import numpy as np
import pytest


class TestSpatialAuto:
    def test_R_zero_unity(self) -> None:
        from naviertwin.core.flow_analysis.two_point import spatial_autocorrelation

        rng = np.random.default_rng(0)
        u = rng.standard_normal((100, 80))
        r, R = spatial_autocorrelation(u, dx=0.05)
        np.testing.assert_allclose(R[0], 1.0, atol=1e-10)

    def test_uncorrelated_decay(self) -> None:
        from naviertwin.core.flow_analysis.two_point import spatial_autocorrelation

        rng = np.random.default_rng(1)
        u = rng.standard_normal((200, 100))
        r, R = spatial_autocorrelation(u, dx=1.0, max_lag=20)
        # 큰 r 에서는 0에 가까움
        assert abs(R[15]) < 0.2
        assert R[0] == 1.0

    def test_correlated_field(self) -> None:
        from naviertwin.core.flow_analysis.two_point import spatial_autocorrelation

        # 인공 공간 상관: 가우시안 변동 + 공간 평균화
        rng = np.random.default_rng(2)
        n_x = 200
        u = np.zeros((100, n_x))
        for t in range(100):
            raw = rng.standard_normal(n_x)
            # 단순 이동 평균으로 상관 도입
            kernel = np.exp(-((np.arange(-5, 6)) ** 2) / 4.0)
            kernel /= kernel.sum()
            u[t] = np.convolve(raw, kernel, mode="same")
        r, R = spatial_autocorrelation(u, dx=1.0, max_lag=20)
        # 짧은 거리에서 강한 양의 상관
        assert R[1] > 0.5

    def test_invalid_ndim_raises(self) -> None:
        from naviertwin.core.flow_analysis.two_point import spatial_autocorrelation

        with pytest.raises(ValueError, match="2D"):
            spatial_autocorrelation(np.zeros(100))

    def test_max_lag_clip(self) -> None:
        from naviertwin.core.flow_analysis.two_point import spatial_autocorrelation

        rng = np.random.default_rng(3)
        u = rng.standard_normal((50, 30))
        r, R = spatial_autocorrelation(u, max_lag=200)
        assert len(R) == 30  # min(200, 30-1) + 1 = 30


class TestTemporalAuto:
    def test_R_zero_unity_1d(self) -> None:
        from naviertwin.core.flow_analysis.two_point import temporal_autocorrelation

        rng = np.random.default_rng(4)
        u = rng.standard_normal(500)
        tau, R = temporal_autocorrelation(u, dt=0.01)
        np.testing.assert_allclose(R[0], 1.0, atol=1e-10)

    def test_R_zero_unity_2d(self) -> None:
        from naviertwin.core.flow_analysis.two_point import temporal_autocorrelation

        rng = np.random.default_rng(5)
        u = rng.standard_normal((300, 10))
        tau, R = temporal_autocorrelation(u, dt=0.1)
        np.testing.assert_allclose(R[0], 1.0, atol=1e-10)

    def test_AR1_process(self) -> None:
        from naviertwin.core.flow_analysis.two_point import temporal_autocorrelation

        # AR(1): u[t] = phi * u[t-1] + noise; 자기상관 R(τ) = phi^τ
        rng = np.random.default_rng(6)
        n = 5000
        phi = 0.7
        u = np.zeros(n)
        for t in range(1, n):
            u[t] = phi * u[t - 1] + rng.standard_normal()
        tau, R = temporal_autocorrelation(u, max_lag=10)
        # R[1] ≈ phi
        assert abs(R[1] - phi) < 0.1

    def test_invalid_ndim_raises(self) -> None:
        from naviertwin.core.flow_analysis.two_point import temporal_autocorrelation

        with pytest.raises(ValueError, match="1D or 2D"):
            temporal_autocorrelation(np.zeros((5, 5, 5)))


class TestIntegralScales:
    def test_length_scale_exponential(self) -> None:
        from naviertwin.core.flow_analysis.two_point import integral_length_scale_from_R

        # R(r) = exp(-r/L₀), L = ∫ exp(-r/L₀) dr = L₀
        L0 = 0.5
        r = np.linspace(0, 5, 200)
        R = np.exp(-r / L0)
        L = integral_length_scale_from_R(r, R)
        np.testing.assert_allclose(L, L0, rtol=0.02)

    def test_length_scale_with_negative_lobe(self) -> None:
        from naviertwin.core.flow_analysis.two_point import integral_length_scale_from_R

        # Cosine envelope; 첫 음수 직전까지만 적분
        r = np.linspace(0, 4 * np.pi, 200)
        R = np.exp(-r / 5.0) * np.cos(r)
        L = integral_length_scale_from_R(r, R)
        assert L > 0
        assert L < r[-1]

    def test_time_scale_alias(self) -> None:
        from naviertwin.core.flow_analysis.two_point import integral_time_scale_from_R

        tau = np.linspace(0, 5, 100)
        R = np.exp(-tau / 0.3)
        T = integral_time_scale_from_R(tau, R)
        np.testing.assert_allclose(T, 0.3, rtol=0.05)

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.two_point import integral_length_scale_from_R

        with pytest.raises(ValueError, match="same-shape"):
            integral_length_scale_from_R(np.zeros(10), np.zeros(20))


class TestTaylorMicroscale:
    def test_known_lambda(self) -> None:
        from naviertwin.core.flow_analysis.two_point import taylor_microscale

        # R(r) = 1 - r²/λ² (이차 영역)
        lam = 0.5
        r = np.linspace(0, 0.4, 50)
        R = 1.0 - (r / lam) ** 2
        lam_est = taylor_microscale(r, R)
        np.testing.assert_allclose(lam_est, lam, rtol=0.02)

    def test_too_few_points_raises(self) -> None:
        from naviertwin.core.flow_analysis.two_point import taylor_microscale

        with pytest.raises(ValueError, match="4 points"):
            taylor_microscale(np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.8, 0.5]))

    def test_increasing_R_raises(self) -> None:
        from naviertwin.core.flow_analysis.two_point import taylor_microscale

        # 비물리: R 증가 → fit 실패
        r = np.linspace(0, 1, 20)
        R = 1 + 0.1 * r ** 2
        with pytest.raises(ValueError, match="fit"):
            taylor_microscale(r, R)
