"""Round 622 — goodness-of-fit tests (KS, AD, χ², Shapiro-Wilk)."""

from __future__ import annotations

import numpy as np
import pytest


class TestKSAgainstCDF:
    def test_uniform_against_uniform_cdf(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import ks_test_against_cdf

        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, 1000)
        D, p = ks_test_against_cdf(x, lambda v: np.clip(v, 0, 1))
        # 같은 분포 → p가 큼
        assert p > 0.05

    def test_normal_against_uniform_low_p(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import ks_test_against_cdf

        rng = np.random.default_rng(1)
        x = rng.standard_normal(500)
        D, p = ks_test_against_cdf(x, lambda v: np.clip((v + 3) / 6, 0, 1))
        # 다른 분포 → p < 0.05
        assert p < 0.05

    def test_too_few_raises(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import ks_test_against_cdf

        with pytest.raises(ValueError, match="2 samples"):
            ks_test_against_cdf(np.array([1.0]), lambda v: v)


class TestKSNormal:
    def test_gaussian_passes(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import ks_test_normal

        rng = np.random.default_rng(2)
        x = rng.standard_normal(2000)
        D, p = ks_test_normal(x)
        assert p > 0.05

    def test_skewed_fails(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import ks_test_normal

        rng = np.random.default_rng(3)
        x = rng.exponential(1.0, 2000)
        D, p = ks_test_normal(x)
        # 지수분포 → 정규 기각
        assert p < 0.01

    def test_explicit_mu_sigma(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import ks_test_normal

        rng = np.random.default_rng(4)
        x = 5.0 + 2.0 * rng.standard_normal(1000)
        D, p = ks_test_normal(x, mu=5.0, sigma=2.0)
        assert p > 0.05

    def test_sigma_zero_raises(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import ks_test_normal

        with pytest.raises(ValueError, match="sigma"):
            ks_test_normal(np.zeros(100), sigma=0.0)


class TestKSTwoSample:
    def test_same_distribution_high_p(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import ks_test_two_sample

        rng = np.random.default_rng(5)
        x = rng.standard_normal(500)
        y = rng.standard_normal(500)
        D, p = ks_test_two_sample(x, y)
        assert p > 0.05

    def test_different_distributions_low_p(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import ks_test_two_sample

        rng = np.random.default_rng(6)
        x = rng.standard_normal(500)
        y = 5.0 + rng.standard_normal(500)
        D, p = ks_test_two_sample(x, y)
        assert p < 0.001

    def test_too_few_raises(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import ks_test_two_sample

        with pytest.raises(ValueError, match="2 elements"):
            ks_test_two_sample(np.array([1.0]), np.array([1.0, 2.0]))


class TestAndersonDarling:
    def test_gaussian_below_threshold(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import (
            anderson_darling_normal,
        )

        rng = np.random.default_rng(7)
        x = rng.standard_normal(500)
        A2, crit = anderson_darling_normal(x)
        # A² < 5% 임계 → 기각 안 함
        assert A2 < crit["5%"] * 2  # tolerance

    def test_skewed_above_threshold(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import (
            anderson_darling_normal,
        )

        rng = np.random.default_rng(8)
        x = rng.exponential(1.0, 500)
        A2, crit = anderson_darling_normal(x)
        # 지수분포 → A² > 1% 임계
        assert A2 > crit["1%"]

    def test_too_few_raises(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import (
            anderson_darling_normal,
        )

        with pytest.raises(ValueError, match="8 samples"):
            anderson_darling_normal(np.zeros(5))

    def test_zero_variance_raises(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import (
            anderson_darling_normal,
        )

        with pytest.raises(ValueError, match="zero"):
            anderson_darling_normal(np.ones(20))


class TestChiSquare:
    def test_perfect_match(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import chi_square_test

        obs = np.array([10, 20, 30, 40])
        chi2, dof = chi_square_test(obs, obs)
        assert abs(chi2) < 1e-10
        assert dof == 3

    def test_nonzero_difference(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import chi_square_test

        obs = np.array([15.0, 25.0, 35.0, 25.0])
        exp = np.array([25.0, 25.0, 25.0, 25.0])
        chi2, dof = chi_square_test(obs, exp)
        assert chi2 > 0
        # (10² + 0² + 10² + 0²) / 25 = 200/25 = 8
        np.testing.assert_allclose(chi2, 8.0)

    def test_shape_mismatch(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import chi_square_test

        with pytest.raises(ValueError, match="shape"):
            chi_square_test(np.array([1, 2, 3]), np.array([1, 2]))

    def test_zero_expected_raises(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import chi_square_test

        with pytest.raises(ValueError, match="expected"):
            chi_square_test(np.array([1, 2, 3]), np.array([1, 0, 3]))


class TestShapiroWilk:
    def test_normal_high_p(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import (
            shapiro_wilk_simplified,
        )

        rng = np.random.default_rng(9)
        x = rng.standard_normal(40)
        W, p = shapiro_wilk_simplified(x)
        assert 0 < W <= 1.0001

    def test_uniform_low_W(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import (
            shapiro_wilk_simplified,
        )

        rng = np.random.default_rng(10)
        x = rng.uniform(-1, 1, 40)
        W_uni, _ = shapiro_wilk_simplified(x)
        x_n = rng.standard_normal(40)
        W_n, _ = shapiro_wilk_simplified(x_n)
        # 정규가 더 높음
        assert W_n >= W_uni - 0.1

    def test_too_few_raises(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import (
            shapiro_wilk_simplified,
        )

        with pytest.raises(ValueError, match="8 samples"):
            shapiro_wilk_simplified(np.zeros(5))

    def test_constant_returns_unity(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import (
            shapiro_wilk_simplified,
        )

        W, p = shapiro_wilk_simplified(np.ones(20))
        assert W == 1.0


class TestNormalCDF:
    def test_zero_returns_half(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import normal_cdf

        result = normal_cdf(np.array([0.0]))
        np.testing.assert_allclose(result, [0.5], atol=1e-10)

    def test_large_positive_one(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import normal_cdf

        result = normal_cdf(np.array([10.0]))
        np.testing.assert_allclose(result, [1.0], atol=1e-10)

    def test_large_negative_zero(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import normal_cdf

        result = normal_cdf(np.array([-10.0]))
        np.testing.assert_allclose(result, [0.0], atol=1e-10)

    def test_with_mu_sigma(self) -> None:
        from naviertwin.core.flow_analysis.goodness_of_fit import normal_cdf

        # μ=5에서 P(X ≤ 5) = 0.5
        result = normal_cdf(np.array([5.0]), mu=5.0, sigma=2.0)
        np.testing.assert_allclose(result, [0.5], atol=1e-10)
