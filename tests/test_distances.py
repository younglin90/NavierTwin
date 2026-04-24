"""Round 43 — 통계 거리 테스트."""

from __future__ import annotations

import numpy as np
import pytest


class TestWasserstein:
    def test_identical_samples_zero(self) -> None:
        from naviertwin.core.validation.distances import wasserstein_1d

        x = np.linspace(0, 1, 100)
        assert wasserstein_1d(x, x) == pytest.approx(0.0, abs=1e-12)

    def test_shifted_distributions(self) -> None:
        from naviertwin.core.validation.distances import wasserstein_1d

        rng = np.random.default_rng(0)
        x = rng.standard_normal(1000)
        y = x + 2.0  # 단순 shift
        d = wasserstein_1d(x, y)
        assert abs(d - 2.0) < 0.1


class TestMMD:
    def test_same_distribution_small(self) -> None:
        from naviertwin.core.validation.distances import mmd_gaussian

        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 2))
        Y = rng.standard_normal((200, 2))
        val = mmd_gaussian(X, Y, sigma=1.0)
        assert val >= 0
        assert val < 0.1

    def test_different_distributions_large(self) -> None:
        from naviertwin.core.validation.distances import mmd_gaussian

        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 2))
        Y = rng.standard_normal((200, 2)) + 3.0
        assert mmd_gaussian(X, Y, sigma=1.0) > 0.1


class TestKLGaussian:
    def test_identical_distributions_zero(self) -> None:
        from naviertwin.core.validation.distances import kl_divergence_gaussian

        mu = np.zeros(2)
        cov = np.eye(2)
        assert kl_divergence_gaussian(mu, cov, mu, cov) == pytest.approx(0.0, abs=1e-12)

    def test_nonzero_between_different(self) -> None:
        from naviertwin.core.validation.distances import kl_divergence_gaussian

        mu1 = np.zeros(2)
        mu2 = np.array([1.0, -1.0])
        cov = np.eye(2)
        d = kl_divergence_gaussian(mu1, cov, mu2, cov)
        # 표준 결과: 0.5 · ||μ_1 - μ_2||² = 0.5 · 2 = 1.0
        assert abs(d - 1.0) < 1e-10
