"""Round 603 — spectral_energy analysis coverage."""

from __future__ import annotations

import numpy as np
import pytest


class TestEnergySpectrum1D:
    def test_basic_shape(self) -> None:
        from naviertwin.core.flow_analysis.spectral_energy import energy_spectrum_1d

        u = np.sin(2 * np.pi * np.linspace(0, 1, 128))
        k, E = energy_spectrum_1d(u)
        assert k.shape == E.shape
        assert len(k) == 65  # rfft: N//2 + 1

    def test_non_negative_energy(self) -> None:
        from naviertwin.core.flow_analysis.spectral_energy import energy_spectrum_1d

        rng = np.random.default_rng(0)
        u = rng.standard_normal(256)
        k, E = energy_spectrum_1d(u, window=True)
        assert np.all(E >= 0)

    def test_no_window(self) -> None:
        from naviertwin.core.flow_analysis.spectral_energy import energy_spectrum_1d

        u = np.zeros(64)
        u[0] = 1.0
        k, E = energy_spectrum_1d(u, window=False)
        assert np.all(E >= 0)

    def test_dx_scales_k(self) -> None:
        from naviertwin.core.flow_analysis.spectral_energy import energy_spectrum_1d

        u = np.ones(128)
        k1, _ = energy_spectrum_1d(u, dx=1.0)
        k2, _ = energy_spectrum_1d(u, dx=2.0)
        np.testing.assert_allclose(k2, k1 / 2, atol=1e-12)

    def test_1d_required(self) -> None:
        from naviertwin.core.flow_analysis.spectral_energy import energy_spectrum_1d

        with pytest.raises(ValueError, match="1D"):
            energy_spectrum_1d(np.zeros((10, 10)))


class TestEnergySpectrum2D:
    def test_basic_shape(self) -> None:
        from naviertwin.core.flow_analysis.spectral_energy import energy_spectrum_2d

        rng = np.random.default_rng(1)
        ux = rng.standard_normal((32, 32))
        uy = rng.standard_normal((32, 32))
        k, E = energy_spectrum_2d(ux, uy)
        assert len(k) == len(E)
        assert np.all(E >= 0)

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.spectral_energy import energy_spectrum_2d

        with pytest.raises(ValueError, match="shape"):
            energy_spectrum_2d(np.zeros((10, 10)), np.zeros((8, 10)))

    def test_1d_raises(self) -> None:
        from naviertwin.core.flow_analysis.spectral_energy import energy_spectrum_2d

        with pytest.raises(ValueError, match="2D"):
            energy_spectrum_2d(np.zeros(10), np.zeros(10))


class TestKolmogorovSlope:
    def test_known_slope(self) -> None:
        from naviertwin.core.flow_analysis.spectral_energy import kolmogorov_slope

        k = np.logspace(0, 2, 100)
        E = k ** (-5 / 3)
        slope, r2 = kolmogorov_slope(k, E)
        assert abs(slope - (-5 / 3)) < 0.01
        assert r2 > 0.999

    def test_with_k_range(self) -> None:
        from naviertwin.core.flow_analysis.spectral_energy import kolmogorov_slope

        k = np.logspace(-1, 3, 200)
        E = k ** (-2.0)
        slope, r2 = kolmogorov_slope(k, E, k_range=(1.0, 100.0))
        assert abs(slope - (-2.0)) < 0.05

    def test_not_enough_points_raises(self) -> None:
        from naviertwin.core.flow_analysis.spectral_energy import kolmogorov_slope

        k = np.array([1.0, 2.0, 3.0])
        E = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="valid data"):
            kolmogorov_slope(k, E)

    def test_returns_float_tuple(self) -> None:
        from naviertwin.core.flow_analysis.spectral_energy import kolmogorov_slope

        k = np.array([1.0, 2.0, 4.0])
        E = np.array([1.0, 0.3, 0.1])
        slope, r2 = kolmogorov_slope(k, E)
        assert isinstance(slope, float)
        assert isinstance(r2, float)


class TestIntegralLengthScale:
    def test_positive_result(self) -> None:
        from naviertwin.core.flow_analysis.spectral_energy import integral_length_scale

        k = np.linspace(0.1, 10, 100)
        E = np.exp(-k)
        L = integral_length_scale(k, E)
        assert L > 0

    def test_zero_energy_returns_zero(self) -> None:
        from naviertwin.core.flow_analysis.spectral_energy import integral_length_scale

        k = np.linspace(0.1, 10, 50)
        E = np.zeros(50)
        L = integral_length_scale(k, E)
        assert L == 0.0

    def test_ignores_zero_wavenumber(self) -> None:
        from naviertwin.core.flow_analysis.spectral_energy import integral_length_scale

        k = np.array([0.0, 1.0, 2.0, 3.0])
        E = np.array([100.0, 1.0, 0.5, 0.2])
        L = integral_length_scale(k, E)
        assert np.isfinite(L)
