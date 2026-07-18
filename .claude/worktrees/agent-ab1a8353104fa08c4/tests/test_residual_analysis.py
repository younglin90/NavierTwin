"""Round 634 — surrogate residual diagnostics: Q-Q, DW, leverage, Cook's D."""

from __future__ import annotations

import numpy as np
import pytest


class TestQQData:
    def test_normal_residuals_close_to_line(self) -> None:
        from naviertwin.core.surrogate.residual_analysis import qq_data

        rng = np.random.default_rng(0)
        r = rng.standard_normal(500)
        sorted_r, theo = qq_data(r)
        assert sorted_r.shape == theo.shape == (500,)
        # 정규 → 강한 상관
        cor = np.corrcoef(sorted_r, theo)[0, 1]
        assert cor > 0.99

    def test_too_few_raises(self) -> None:
        from naviertwin.core.surrogate.residual_analysis import qq_data

        with pytest.raises(ValueError, match="2 residuals"):
            qq_data(np.array([1.0]))

    def test_returns_sorted(self) -> None:
        from naviertwin.core.surrogate.residual_analysis import qq_data

        rng = np.random.default_rng(1)
        r = rng.standard_normal(100)
        sorted_r, _ = qq_data(r)
        assert np.all(np.diff(sorted_r) >= 0)


class TestResidualAutocorrelation:
    def test_iid_low_autocorr(self) -> None:
        from naviertwin.core.surrogate.residual_analysis import (
            residual_autocorrelation,
        )

        rng = np.random.default_rng(2)
        r = rng.standard_normal(2000)
        acf = residual_autocorrelation(r, max_lag=20)
        assert acf[0] == 1.0
        # 큰 lag에서 0에 가까움
        assert abs(acf[10]) < 0.1

    def test_short_returns_unity(self) -> None:
        from naviertwin.core.surrogate.residual_analysis import (
            residual_autocorrelation,
        )

        acf = residual_autocorrelation(np.array([1.0]))
        assert acf[0] == 1.0

    def test_AR1_strong(self) -> None:
        from naviertwin.core.surrogate.residual_analysis import (
            residual_autocorrelation,
        )

        rng = np.random.default_rng(3)
        n = 5000
        r = np.zeros(n)
        for t in range(1, n):
            r[t] = 0.8 * r[t - 1] + rng.standard_normal()
        acf = residual_autocorrelation(r, max_lag=5)
        # acf[1] ≈ 0.8
        assert abs(acf[1] - 0.8) < 0.1


class TestDurbinWatson:
    def test_iid_close_to_2(self) -> None:
        from naviertwin.core.surrogate.residual_analysis import durbin_watson

        rng = np.random.default_rng(4)
        r = rng.standard_normal(1000)
        dw = durbin_watson(r)
        assert 1.5 < dw < 2.5

    def test_AR1_below_2(self) -> None:
        from naviertwin.core.surrogate.residual_analysis import durbin_watson

        rng = np.random.default_rng(5)
        n = 1000
        r = np.zeros(n)
        for t in range(1, n):
            r[t] = 0.9 * r[t - 1] + rng.standard_normal()
        dw = durbin_watson(r)
        # 강한 양의 자기상관 → DW < 1
        assert dw < 1.0

    def test_too_short_raises(self) -> None:
        from naviertwin.core.surrogate.residual_analysis import durbin_watson

        with pytest.raises(ValueError, match="2 residuals"):
            durbin_watson(np.array([1.0]))


class TestLeverage:
    def test_average_p_over_n(self) -> None:
        from naviertwin.core.surrogate.residual_analysis import leverage_scores

        rng = np.random.default_rng(6)
        n, p = 50, 5
        X = rng.standard_normal((n, p))
        h = leverage_scores(X)
        # mean(h) = p/n
        np.testing.assert_allclose(h.mean(), p / n, atol=1e-10)

    def test_in_range(self) -> None:
        from naviertwin.core.surrogate.residual_analysis import leverage_scores

        rng = np.random.default_rng(7)
        X = rng.standard_normal((30, 4))
        h = leverage_scores(X)
        assert np.all((0 <= h) & (h <= 1))

    def test_invalid_ndim(self) -> None:
        from naviertwin.core.surrogate.residual_analysis import leverage_scores

        with pytest.raises(ValueError, match="2D"):
            leverage_scores(np.zeros(10))


class TestCooksDistance:
    def test_zero_residual_zero_distance(self) -> None:
        from naviertwin.core.surrogate.residual_analysis import cooks_distance

        D = cooks_distance(
            residuals=np.zeros(5),
            leverage=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            mse=1.0,
            p=2,
        )
        np.testing.assert_array_equal(D, np.zeros(5))

    def test_high_leverage_high_D(self) -> None:
        from naviertwin.core.surrogate.residual_analysis import cooks_distance

        # 같은 잔차, leverage만 다름
        r = np.array([1.0, 1.0])
        h = np.array([0.1, 0.9])
        D = cooks_distance(r, h, mse=1.0, p=2)
        assert D[1] > D[0]

    def test_invalid_p_raises(self) -> None:
        from naviertwin.core.surrogate.residual_analysis import cooks_distance

        with pytest.raises(ValueError, match="p"):
            cooks_distance(np.zeros(5), np.zeros(5), mse=1.0, p=0)

    def test_invalid_mse_raises(self) -> None:
        from naviertwin.core.surrogate.residual_analysis import cooks_distance

        with pytest.raises(ValueError, match="mse"):
            cooks_distance(np.zeros(5), np.zeros(5), mse=0.0, p=2)

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.surrogate.residual_analysis import cooks_distance

        with pytest.raises(ValueError, match="shape"):
            cooks_distance(np.zeros(5), np.zeros(4), mse=1.0, p=2)


class TestShapiroDiagnostic:
    def test_normal_skew_near_zero(self) -> None:
        from naviertwin.core.surrogate.residual_analysis import (
            shapiro_normality_diagnostic,
        )

        rng = np.random.default_rng(8)
        r = rng.standard_normal(5000)
        d = shapiro_normality_diagnostic(r)
        assert abs(d["skewness"]) < 0.2
        assert abs(d["kurtosis"]) < 0.5  # excess

    def test_short_returns_defaults(self) -> None:
        from naviertwin.core.surrogate.residual_analysis import (
            shapiro_normality_diagnostic,
        )

        d = shapiro_normality_diagnostic(np.array([1.0]))
        assert d["mean"] == 0.0
        assert d["dw"] == 2.0

    def test_returns_dict_keys(self) -> None:
        from naviertwin.core.surrogate.residual_analysis import (
            shapiro_normality_diagnostic,
        )

        rng = np.random.default_rng(9)
        d = shapiro_normality_diagnostic(rng.standard_normal(100))
        for k in ("mean", "std", "skewness", "kurtosis", "dw"):
            assert k in d
