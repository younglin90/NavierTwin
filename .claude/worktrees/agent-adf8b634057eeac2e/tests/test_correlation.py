"""Round 91 — 상관 분석."""

from __future__ import annotations

import numpy as np
import pytest


class TestCorrelation:
    def test_pearson_identity(self) -> None:
        from naviertwin.core.analysis.correlation import pearson_matrix

        X = np.vstack([np.arange(50, dtype=float), np.arange(50, dtype=float)])
        C = pearson_matrix(X)
        assert C.shape == (2, 2)
        assert C[0, 1] == pytest.approx(1.0)

    def test_pearson_anti(self) -> None:
        from naviertwin.core.analysis.correlation import pearson_matrix

        X = np.vstack([np.arange(20, dtype=float), -np.arange(20, dtype=float)])
        C = pearson_matrix(X)
        assert C[0, 1] == pytest.approx(-1.0)

    def test_spearman_monotonic(self) -> None:
        from naviertwin.core.analysis.correlation import spearman_matrix

        # 비선형 monotonic → Spearman=1
        x = np.arange(1, 11, dtype=float)
        y = x ** 3
        S = spearman_matrix(np.vstack([x, y]))
        assert S[0, 1] == pytest.approx(1.0)

    def test_cross_correlation_lag(self) -> None:
        from naviertwin.core.analysis.correlation import cross_correlation

        rng = np.random.default_rng(0)
        a = rng.standard_normal(500)
        b = np.roll(a, 5)  # b = a 시프트
        lags, corr = cross_correlation(a, b, max_lag=20)
        peak_lag = int(lags[np.argmax(corr)])
        assert abs(peak_lag) == 5  # ±5 (convention)

    def test_cross_corr_len_mismatch(self) -> None:
        from naviertwin.core.analysis.correlation import cross_correlation

        with pytest.raises(ValueError):
            cross_correlation(np.zeros(5), np.zeros(7))
