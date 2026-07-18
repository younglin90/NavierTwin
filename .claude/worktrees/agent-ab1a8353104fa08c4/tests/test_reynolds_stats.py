"""Round 606 — Reynolds decomposition + turbulence statistics."""

from __future__ import annotations

import numpy as np
import pytest


class TestReynoldsBasics:
    def test_mean_field(self) -> None:
        from naviertwin.core.flow_analysis.reynolds_stats import mean_field

        u = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        m = mean_field(u, axis=0)
        np.testing.assert_allclose(m, [3.0, 4.0])

    def test_fluctuation_zero_mean(self) -> None:
        from naviertwin.core.flow_analysis.reynolds_stats import fluctuation

        rng = np.random.default_rng(0)
        u = 5.0 + rng.standard_normal((100, 20))
        up = fluctuation(u, axis=0)
        np.testing.assert_allclose(up.mean(axis=0), 0.0, atol=1e-12)

    def test_rms_shape(self) -> None:
        from naviertwin.core.flow_analysis.reynolds_stats import rms

        rng = np.random.default_rng(1)
        u = rng.standard_normal((50, 10))
        r = rms(u, axis=0)
        assert r.shape == (10,)
        assert np.all(r > 0)

    def test_rms_constant_zero(self) -> None:
        from naviertwin.core.flow_analysis.reynolds_stats import rms

        u = np.ones((20, 5))
        r = rms(u)
        np.testing.assert_allclose(r, 0.0)


class TestReynoldsStress:
    def test_2d_components(self) -> None:
        from naviertwin.core.flow_analysis.reynolds_stats import reynolds_stress_2d

        rng = np.random.default_rng(2)
        u = rng.standard_normal((200, 8))
        v = rng.standard_normal((200, 8))
        rs = reynolds_stress_2d(u, v)
        assert "uu" in rs and "vv" in rs and "uv" in rs
        assert rs["uu"].shape == (8,)
        assert np.all(rs["uu"] > 0)
        assert np.all(rs["vv"] > 0)

    def test_2d_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.reynolds_stats import reynolds_stress_2d

        with pytest.raises(ValueError, match="shapes"):
            reynolds_stress_2d(np.zeros((10, 5)), np.zeros((10, 6)))

    def test_3d_components(self) -> None:
        from naviertwin.core.flow_analysis.reynolds_stats import reynolds_stress_3d

        rng = np.random.default_rng(3)
        u = rng.standard_normal((100, 6))
        v = rng.standard_normal((100, 6))
        w = rng.standard_normal((100, 6))
        rs = reynolds_stress_3d(u, v, w)
        for key in ("uu", "vv", "ww", "uv", "uw", "vw"):
            assert key in rs

    def test_3d_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.reynolds_stats import reynolds_stress_3d

        with pytest.raises(ValueError, match="shapes"):
            reynolds_stress_3d(
                np.zeros((10, 5)),
                np.zeros((10, 5)),
                np.zeros((10, 4)),
            )


class TestTKEAndIntensity:
    def test_tke_2d(self) -> None:
        from naviertwin.core.flow_analysis.reynolds_stats import (
            turbulent_kinetic_energy,
        )

        rng = np.random.default_rng(4)
        u = rng.standard_normal((100, 5))
        v = rng.standard_normal((100, 5))
        k = turbulent_kinetic_energy(u, v)
        assert k.shape == (5,)
        assert np.all(k >= 0)

    def test_tke_3d(self) -> None:
        from naviertwin.core.flow_analysis.reynolds_stats import (
            turbulent_kinetic_energy,
        )

        rng = np.random.default_rng(5)
        u = rng.standard_normal((100, 5))
        v = rng.standard_normal((100, 5))
        w = rng.standard_normal((100, 5))
        k = turbulent_kinetic_energy(u, v, w)
        assert k.shape == (5,)

    def test_intensity_1d(self) -> None:
        from naviertwin.core.flow_analysis.reynolds_stats import turbulence_intensity

        rng = np.random.default_rng(6)
        u = 5.0 + 0.1 * rng.standard_normal((300, 10))
        ti = turbulence_intensity(u)
        assert ti.shape == (10,)
        assert np.all(ti < 0.1)  # 작은 변동

    def test_intensity_3d(self) -> None:
        from naviertwin.core.flow_analysis.reynolds_stats import turbulence_intensity

        rng = np.random.default_rng(7)
        u = 1.0 + 0.05 * rng.standard_normal((300, 8))
        v = 0.5 + 0.05 * rng.standard_normal((300, 8))
        w = 0.0 + 0.05 * rng.standard_normal((300, 8))
        ti = turbulence_intensity(u, v, w)
        assert ti.shape == (8,)
        assert np.all(ti > 0)


class TestStatistics:
    def test_skewness_normal_near_zero(self) -> None:
        from naviertwin.core.flow_analysis.reynolds_stats import skewness

        rng = np.random.default_rng(8)
        u = rng.standard_normal((10000, 4))
        s = skewness(u)
        assert s.shape == (4,)
        assert np.all(np.abs(s) < 0.2)

    def test_flatness_normal_near_three(self) -> None:
        from naviertwin.core.flow_analysis.reynolds_stats import flatness

        rng = np.random.default_rng(9)
        u = rng.standard_normal((10000, 4))
        f = flatness(u)
        assert f.shape == (4,)
        assert np.all(np.abs(f - 3.0) < 0.5)


class TestRunningStatistics:
    def test_running_mean_converges(self) -> None:
        from naviertwin.core.flow_analysis.reynolds_stats import running_statistics

        rng = np.random.default_rng(10)
        u = 7.0 + rng.standard_normal((500, 6))
        stats = running_statistics(u)
        assert stats["mean"].shape == u.shape
        # 마지막 시점 평균 ≈ 7
        np.testing.assert_allclose(stats["mean"][-1], 7.0, atol=0.2)

    def test_running_rms_initialized(self) -> None:
        from naviertwin.core.flow_analysis.reynolds_stats import running_statistics

        rng = np.random.default_rng(11)
        u = rng.standard_normal((300, 4))
        stats = running_statistics(u)
        # t=0에는 RMS=0
        np.testing.assert_allclose(stats["rms"][0], 0.0)
        # 마지막 시점 RMS ≈ 1
        np.testing.assert_allclose(stats["rms"][-1], 1.0, atol=0.2)

    def test_running_invalid_axis_raises(self) -> None:
        from naviertwin.core.flow_analysis.reynolds_stats import running_statistics

        with pytest.raises(ValueError, match="axis"):
            running_statistics(np.zeros((10, 5)), axis=1)

    def test_running_empty_returns_zero(self) -> None:
        from naviertwin.core.flow_analysis.reynolds_stats import running_statistics

        u = np.zeros((0, 5))
        stats = running_statistics(u)
        assert stats["mean"].shape == (0, 5)
