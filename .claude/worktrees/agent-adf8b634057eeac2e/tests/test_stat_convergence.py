"""Round 611 — statistical convergence diagnostics."""

from __future__ import annotations

import numpy as np
import pytest


class TestBatchMeans:
    def test_iid_normal(self) -> None:
        from naviertwin.core.flow_analysis.stat_convergence import batch_means_se

        rng = np.random.default_rng(0)
        x = 5.0 + rng.standard_normal(10000)
        mean, se = batch_means_se(x, n_batches=20)
        assert abs(mean - 5.0) < 0.1
        assert se > 0
        assert se < 0.5

    def test_invalid_batches_raises(self) -> None:
        from naviertwin.core.flow_analysis.stat_convergence import batch_means_se

        with pytest.raises(ValueError, match="n_batches"):
            batch_means_se(np.zeros(100), n_batches=1)

    def test_too_short_raises(self) -> None:
        from naviertwin.core.flow_analysis.stat_convergence import batch_means_se

        with pytest.raises(ValueError, match="too short"):
            batch_means_se(np.zeros(10), n_batches=20)


class TestGeweke:
    def test_converged_low_z(self) -> None:
        from naviertwin.core.flow_analysis.stat_convergence import geweke_diagnostic

        rng = np.random.default_rng(1)
        x = rng.standard_normal(10000)
        z = geweke_diagnostic(x)
        assert abs(z) < 3.0

    def test_drift_high_z(self) -> None:
        from naviertwin.core.flow_analysis.stat_convergence import geweke_diagnostic

        rng = np.random.default_rng(2)
        # 평균이 시간에 따라 변함 (drift)
        n = 5000
        x = np.linspace(0, 5, n) + 0.1 * rng.standard_normal(n)
        z = geweke_diagnostic(x)
        assert abs(z) > 5.0

    def test_invalid_frac_raises(self) -> None:
        from naviertwin.core.flow_analysis.stat_convergence import geweke_diagnostic

        with pytest.raises(ValueError, match="frac"):
            geweke_diagnostic(np.zeros(100), first_frac=1.5)

    def test_overlapping_frac_raises(self) -> None:
        from naviertwin.core.flow_analysis.stat_convergence import geweke_diagnostic

        with pytest.raises(ValueError, match="<= 1"):
            geweke_diagnostic(np.zeros(100), first_frac=0.6, last_frac=0.6)


class TestEffectiveSampleSize:
    def test_iid_close_to_N(self) -> None:
        from naviertwin.core.flow_analysis.stat_convergence import (
            effective_sample_size,
        )

        rng = np.random.default_rng(3)
        x = rng.standard_normal(2000)
        ess = effective_sample_size(x)
        # IID → ESS ≈ N
        assert ess > 0.7 * 2000

    def test_AR1_smaller_ess(self) -> None:
        from naviertwin.core.flow_analysis.stat_convergence import (
            effective_sample_size,
        )

        rng = np.random.default_rng(4)
        n = 5000
        phi = 0.9
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + rng.standard_normal()
        ess = effective_sample_size(x)
        # 강한 자기상관 → ESS << N
        assert ess < n / 2

    def test_short_series(self) -> None:
        from naviertwin.core.flow_analysis.stat_convergence import (
            effective_sample_size,
        )

        ess = effective_sample_size(np.array([1.0]))
        assert ess == 1.0


class TestPlateauDetector:
    def test_converged_signal_finds_plateau(self) -> None:
        from naviertwin.core.flow_analysis.stat_convergence import plateau_detector

        rng = np.random.default_rng(5)
        x = 1.0 + 0.05 * rng.standard_normal(2000)
        idx = plateau_detector(x, window=200, tol_rel=0.01)
        assert idx is not None
        assert idx >= 0

    def test_drift_returns_none(self) -> None:
        from naviertwin.core.flow_analysis.stat_convergence import plateau_detector

        x = np.linspace(0, 10, 1000)  # 지속적 drift
        idx = plateau_detector(x, window=100, tol_rel=0.001)
        # 강한 drift → plateau 못 찾음
        assert idx is None or idx > 800

    def test_short_series_returns_none(self) -> None:
        from naviertwin.core.flow_analysis.stat_convergence import plateau_detector

        idx = plateau_detector(np.zeros(50), window=100)
        assert idx is None

    def test_invalid_window_raises(self) -> None:
        from naviertwin.core.flow_analysis.stat_convergence import plateau_detector

        with pytest.raises(ValueError, match="window"):
            plateau_detector(np.zeros(100), window=0)


class TestAutocorrTime:
    def test_iid_close_to_one(self) -> None:
        from naviertwin.core.flow_analysis.stat_convergence import (
            autocorrelation_time,
        )

        rng = np.random.default_rng(6)
        x = rng.standard_normal(3000)
        tau = autocorrelation_time(x)
        # IID → τ ≈ 1
        assert 0.5 < tau < 3.0

    def test_AR1_larger(self) -> None:
        from naviertwin.core.flow_analysis.stat_convergence import (
            autocorrelation_time,
        )

        rng = np.random.default_rng(7)
        n = 5000
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.9 * x[t - 1] + rng.standard_normal()
        tau = autocorrelation_time(x)
        # AR(1) phi=0.9 → τ ≈ (1+phi)/(1-phi) = 19
        assert tau > 5.0

    def test_short_returns_one(self) -> None:
        from naviertwin.core.flow_analysis.stat_convergence import (
            autocorrelation_time,
        )

        tau = autocorrelation_time(np.array([1.0]))
        assert tau == 1.0
