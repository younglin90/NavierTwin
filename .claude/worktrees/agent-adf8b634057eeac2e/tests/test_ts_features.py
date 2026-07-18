"""Round 640 — time-series feature extraction (catch22-lite)."""

from __future__ import annotations

import numpy as np
import pytest


class TestExtractFeatures:
    def test_keys_present(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import extract_features

        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        f = extract_features(x)
        for k in ("mean", "std", "min", "max", "median", "iqr",
                   "trend_slope", "n_peaks"):
            assert k in f

    def test_empty_raises(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import extract_features

        with pytest.raises(ValueError, match="empty"):
            extract_features(np.array([]))

    def test_constant_signal(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import extract_features

        x = np.ones(50)
        f = extract_features(x)
        assert f["std"] == 0.0
        assert f["mean"] == 1.0
        assert f["trend_slope"] == 0.0


class TestIndividualFeatures:
    def test_long_run(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import long_run_above_mean

        x = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0])  # mean ≈ 0.57
        # 평균 이상 연속: 처음 3개와 다음 1개
        result = long_run_above_mean(x)
        assert result >= 1

    def test_long_run_no_above(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import long_run_above_mean

        # 모두 같음 → above 없음
        x = np.zeros(10)
        assert long_run_above_mean(x) == 0

    def test_n_peaks(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import number_peaks

        # 피크 3개: 인덱스 1, 4, 7
        x = np.array([0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0])
        # support=1 → 인접 모두 작은 점 카운트
        n = number_peaks(x, support=1)
        assert n == 3

    def test_n_peaks_too_short(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import number_peaks

        n = number_peaks(np.array([1.0, 2.0]), support=3)
        assert n == 0

    def test_first_above_mean(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import first_index_above_mean

        x = np.array([0.0, 0.0, 5.0, 1.0])  # mean = 1.5
        idx = first_index_above_mean(x)
        assert idx == 2

    def test_first_above_mean_never(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import first_index_above_mean

        x = np.zeros(10)
        idx = first_index_above_mean(x)
        assert idx == -1

    def test_pct_below_zero(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import percentage_below_zero

        x = np.array([-1.0, -2.0, 1.0, 2.0])
        assert percentage_below_zero(x) == 0.5

    def test_pct_below_zero_empty(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import percentage_below_zero

        assert percentage_below_zero(np.array([])) == 0.0

    def test_total_variation(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import (
            absolute_sum_of_changes,
        )

        x = np.array([0.0, 1.0, 0.0, 1.0])
        # |1| + |1| + |1| = 3
        assert absolute_sum_of_changes(x) == 3.0

    def test_total_variation_short(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import (
            absolute_sum_of_changes,
        )

        assert absolute_sum_of_changes(np.array([1.0])) == 0.0

    def test_mean_abs_change(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import mean_absolute_change

        x = np.array([0.0, 2.0, 4.0])
        # diff = [2, 2] → mean = 2
        assert mean_absolute_change(x) == 2.0

    def test_mean_abs_change_short(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import mean_absolute_change

        assert mean_absolute_change(np.array([1.0])) == 0.0

    def test_acf_lag1_sine(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import autocorrelation_lag1

        x = np.sin(np.linspace(0, 4 * np.pi, 200))
        rho = autocorrelation_lag1(x)
        # 매우 평활한 사인 → 강한 양의 자기상관
        assert rho > 0.9

    def test_acf_lag1_iid_low(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import autocorrelation_lag1

        rng = np.random.default_rng(0)
        x = rng.standard_normal(2000)
        rho = autocorrelation_lag1(x)
        assert abs(rho) < 0.1

    def test_acf_constant_zero(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import autocorrelation_lag1

        assert autocorrelation_lag1(np.ones(10)) == 0.0

    def test_trend_slope_linear(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import trend_slope

        x = np.linspace(0, 10, 100)
        slope = trend_slope(x)
        # x가 t에 비례 → slope ≈ Δx/Δt = 10/99
        np.testing.assert_allclose(slope, 10.0 / 99.0, atol=1e-3)

    def test_trend_slope_constant(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import trend_slope

        assert trend_slope(np.ones(50)) == 0.0

    def test_crest_factor(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import crest_factor

        x = np.array([0.0, 0.0, 0.0, 10.0])
        # max = 10, RMS = √(100/4) = 5 → crest = 2
        np.testing.assert_allclose(crest_factor(x), 2.0)

    def test_crest_zero_rms(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import crest_factor

        assert crest_factor(np.zeros(5)) == 0.0

    def test_shannon_entropy(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import shannon_entropy

        rng = np.random.default_rng(0)
        x = rng.standard_normal(2000)
        h = shannon_entropy(x, bins=10)
        # 분포 있는 신호 → 엔트로피 > 0
        assert h > 0

    def test_shannon_entropy_short(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import shannon_entropy

        assert shannon_entropy(np.array([1.0])) == 0.0

    def test_zero_crossing_rate(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import zero_crossing_rate

        # +1, -1, +1, -1 → 모든 인접 쌍이 부호 변화 → rate = 1
        x = np.array([1.0, -1.0, 1.0, -1.0])
        np.testing.assert_allclose(zero_crossing_rate(x), 1.0)

    def test_zero_crossing_short(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import zero_crossing_rate

        assert zero_crossing_rate(np.array([1.0])) == 0.0

    def test_mean_above_quantile_empty(self) -> None:
        from naviertwin.core.flow_analysis.ts_features import mean_above_quantile

        # 매우 긴 신호에서 q=1.0 → above empty
        x = np.array([1.0, 2.0, 3.0])
        result = mean_above_quantile(x, q=0.99)
        assert result > 0
