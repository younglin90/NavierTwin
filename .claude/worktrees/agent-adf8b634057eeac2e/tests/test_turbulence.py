"""Round 36 — 에너지 스펙트럼 + k-ε closure."""

from __future__ import annotations

import numpy as np
import pytest


class TestEnergySpectrum:
    def test_1d_spectrum(self) -> None:
        from naviertwin.core.turbulence.energy_spectrum import energy_spectrum_1d

        N = 256
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        u = np.sin(5 * x)  # 단일 파수
        k, E = energy_spectrum_1d(u)
        # k=5 에서 피크
        assert np.argmax(E) == 5

    def test_2d_spectrum_shape(self) -> None:
        from naviertwin.core.turbulence.energy_spectrum import energy_spectrum_2d

        rng = np.random.default_rng(0)
        u = rng.standard_normal((32, 32))
        v = rng.standard_normal((32, 32))
        k, E = energy_spectrum_2d(u, v)
        assert k.shape == E.shape
        assert np.all(E >= 0)

    def test_kolmogorov_slope_sign(self) -> None:
        """-5/3 처럼 음의 기울기가 나오도록 합성."""
        from naviertwin.core.turbulence.energy_spectrum import (
            energy_spectrum_1d,
            kolmogorov_slope,
        )

        rng = np.random.default_rng(0)
        N = 512
        # k^(-5/6) 랜덤 위상 spectrum
        U = np.zeros(N // 2 + 1, dtype=complex)
        for k in range(1, U.size):
            amp = k ** (-5 / 6)
            phi = rng.random() * 2 * np.pi
            U[k] = amp * np.exp(1j * phi)
        u = np.fft.irfft(U, n=N)
        k, E = energy_spectrum_1d(u)
        slope = kolmogorov_slope(k, E, k_min_ratio=0.1, k_max_ratio=0.5)
        # Kolmogorov -5/3 근처 — power 2배하면 slope ≈ -5/3
        assert slope < 0


class TestKEpsilon:
    def test_eddy_viscosity(self) -> None:
        from naviertwin.core.turbulence.k_epsilon import eddy_viscosity

        k = 0.5 * np.ones((5, 5))
        eps = 0.1 * np.ones((5, 5))
        nu_t = eddy_viscosity(k, eps)
        # C_μ · k² / ε = 0.09 · 0.25 / 0.1 = 0.225
        assert np.allclose(nu_t, 0.225, atol=1e-6)

    def test_negative_raises(self) -> None:
        from naviertwin.core.turbulence.k_epsilon import eddy_viscosity

        with pytest.raises(ValueError):
            eddy_viscosity(np.array([-1.0]), np.array([0.1]))

    def test_step_positive(self) -> None:
        from naviertwin.core.turbulence.k_epsilon import k_epsilon_step

        rng = np.random.default_rng(0)
        k = 0.5 + 0.01 * rng.standard_normal((10, 10))
        eps = 0.1 + 0.01 * rng.standard_normal((10, 10))
        u = rng.standard_normal((10, 10))
        v = rng.standard_normal((10, 10))
        k_new, eps_new = k_epsilon_step(
            k, eps, u, v, dt=0.001, dx=0.1, dy=0.1,
        )
        assert np.all(k_new > 0)
        assert np.all(eps_new > 0)
