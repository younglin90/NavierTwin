"""Round 620 — quantile statistics + box stats + ECDF + robust means."""

from __future__ import annotations

import numpy as np
import pytest


class TestPercentile:
    def test_scalar(self) -> None:
        from naviertwin.core.flow_analysis.quantile_stats import percentile

        x = np.arange(101.0)
        assert abs(percentile(x, 50) - 50.0) < 1e-12
        assert abs(percentile(x, 0) - 0.0) < 1e-12
        assert abs(percentile(x, 100) - 100.0) < 1e-12

    def test_array_p(self) -> None:
        from naviertwin.core.flow_analysis.quantile_stats import percentile

        x = np.arange(101.0)
        result = percentile(x, np.array([25.0, 50.0, 75.0]))
        np.testing.assert_allclose(result, [25, 50, 75])

    def test_invalid_p_raises(self) -> None:
        from naviertwin.core.flow_analysis.quantile_stats import percentile

        with pytest.raises(ValueError, match="0, 100"):
            percentile(np.zeros(10), 150)


class TestQuartilesIQR:
    def test_quartiles(self) -> None:
        from naviertwin.core.flow_analysis.quantile_stats import quartiles

        x = np.arange(101.0)
        q1, q2, q3 = quartiles(x)
        np.testing.assert_allclose([q1, q2, q3], [25, 50, 75])

    def test_iqr(self) -> None:
        from naviertwin.core.flow_analysis.quantile_stats import iqr

        x = np.arange(101.0)
        np.testing.assert_allclose(iqr(x), 50.0)


class TestBoxStats:
    def test_basic_dict(self) -> None:
        from naviertwin.core.flow_analysis.quantile_stats import box_stats

        rng = np.random.default_rng(0)
        x = rng.standard_normal(1000)
        s = box_stats(x)
        for k in ("median", "Q1", "Q3", "iqr", "whisker_low",
                   "whisker_high", "n_outliers", "min", "max"):
            assert k in s

    def test_outliers_detected(self) -> None:
        from naviertwin.core.flow_analysis.quantile_stats import box_stats

        rng = np.random.default_rng(1)
        x = rng.standard_normal(1000)
        # 강한 spike 추가
        x = np.concatenate([x, [100.0, -100.0, 50.0, -50.0]])
        s = box_stats(x, whisker_factor=1.5)
        assert s["n_outliers"] >= 4

    def test_invalid_whisker_raises(self) -> None:
        from naviertwin.core.flow_analysis.quantile_stats import box_stats

        with pytest.raises(ValueError, match="whisker_factor"):
            box_stats(np.zeros(100), whisker_factor=-0.5)


class TestECDF:
    def test_monotonic(self) -> None:
        from naviertwin.core.flow_analysis.quantile_stats import empirical_cdf

        rng = np.random.default_rng(2)
        x = rng.standard_normal(500)
        s, F = empirical_cdf(x)
        assert np.all(np.diff(s) >= 0)
        assert np.all(np.diff(F) > 0)
        assert F[0] > 0 and F[-1] < 1

    def test_uniform_dist(self) -> None:
        from naviertwin.core.flow_analysis.quantile_stats import empirical_cdf

        x = np.linspace(0, 1, 1000)
        s, F = empirical_cdf(x)
        # F(0.5) ≈ 0.5
        idx = np.argmin(np.abs(s - 0.5))
        assert abs(F[idx] - 0.5) < 0.01


class TestOutliersIQR:
    def test_clean_data(self) -> None:
        from naviertwin.core.flow_analysis.quantile_stats import outliers_iqr

        rng = np.random.default_rng(3)
        x = rng.standard_normal(1000)
        mask = outliers_iqr(x)
        # 보통 < 1% outlier
        assert mask.sum() < 50

    def test_with_outliers(self) -> None:
        from naviertwin.core.flow_analysis.quantile_stats import outliers_iqr

        x = np.concatenate([np.zeros(100), [50.0, -50.0]])
        mask = outliers_iqr(x)
        # 마지막 두 점이 outlier
        assert mask[-1] and mask[-2]


class TestTrimmedMean:
    def test_basic(self) -> None:
        from naviertwin.core.flow_analysis.quantile_stats import trimmed_mean

        x = np.arange(101.0)
        # 10% trim → 양 끝 10개씩 제거 → 중앙 81개의 평균 = 50
        m = trimmed_mean(x, trim_fraction=0.1)
        np.testing.assert_allclose(m, 50.0)

    def test_robust_to_outliers(self) -> None:
        from naviertwin.core.flow_analysis.quantile_stats import trimmed_mean

        rng = np.random.default_rng(4)
        x = np.concatenate([rng.standard_normal(100), [1000.0, -1000.0]])
        # trimmed mean은 outlier에 둔감
        tm = trimmed_mean(x, trim_fraction=0.05)
        assert abs(tm) < 0.5

    def test_invalid_trim(self) -> None:
        from naviertwin.core.flow_analysis.quantile_stats import trimmed_mean

        with pytest.raises(ValueError, match="trim_fraction"):
            trimmed_mean(np.zeros(100), trim_fraction=0.5)

    def test_full_trim_returns_median(self) -> None:
        from naviertwin.core.flow_analysis.quantile_stats import trimmed_mean

        x = np.arange(11.0)
        # trim = 0.49 → cut = 5 → 모두 제거 → median 반환
        m = trimmed_mean(x, trim_fraction=0.49)
        np.testing.assert_allclose(m, 5.0)


class TestWinsorized:
    def test_basic(self) -> None:
        from naviertwin.core.flow_analysis.quantile_stats import winsorized_mean

        x = np.arange(101.0)
        # 5% winsorize → 양 끝 5개를 인접 값으로 클립
        m = winsorized_mean(x, limits=(0.05, 0.05))
        # winsorize는 평균 변화 적음 (대칭이라)
        np.testing.assert_allclose(m, 50.0, atol=2.0)

    def test_invalid_limits_raises(self) -> None:
        from naviertwin.core.flow_analysis.quantile_stats import winsorized_mean

        with pytest.raises(ValueError, match="limits"):
            winsorized_mean(np.zeros(100), limits=(0.6, 0.1))

    def test_zero_limits_returns_mean(self) -> None:
        from naviertwin.core.flow_analysis.quantile_stats import winsorized_mean

        x = np.arange(11.0)
        m = winsorized_mean(x, limits=(0.0, 0.0))
        np.testing.assert_allclose(m, x.mean())
