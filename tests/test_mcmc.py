"""Round 181 — MH MCMC."""

from __future__ import annotations

import numpy as np


class TestMCMC:
    def test_gaussian_moments(self) -> None:
        """N(0, 1) 로부터 샘플 → 평균/분산 ≈ 0, 1."""
        from naviertwin.core.uncertainty.mcmc import metropolis_hastings

        def logp(x):
            return -0.5 * float(x[0]) ** 2

        s = metropolis_hastings(logp, np.zeros(1), n=8000, step=1.0, seed=0)
        assert abs(float(s.mean())) < 0.1
        assert abs(float(s.var()) - 1.0) < 0.2

    def test_banana(self) -> None:
        """간단한 2D 비정규: p(x,y) ∝ exp(-(y - x²)² - x²/4)."""
        from naviertwin.core.uncertainty.mcmc import metropolis_hastings

        def logp(z):
            x, y = z
            return -((y - x ** 2) ** 2) - 0.25 * x ** 2

        s = metropolis_hastings(logp, np.zeros(2), n=5000, step=0.8, seed=0)
        # 주변 분포 유효한 지 체크
        assert np.all(np.isfinite(s))
        # x² 의 평균은 0.5 (from -x²/4 Gaussian → variance=2)
        assert abs(s[:, 0].mean()) < 0.5

    def test_acceptance(self) -> None:
        from naviertwin.core.uncertainty.mcmc import acceptance_rate

        r = acceptance_rate(
            lambda x: -0.5 * float(x[0]) ** 2, np.zeros(1),
            n=1000, step=1.0, seed=0,
        )
        assert 0.1 < r < 0.9
