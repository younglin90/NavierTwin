"""Round 321 — SMC."""

from __future__ import annotations

import numpy as np


class TestSMC:
    def test_gaussian_target_mean_zero(self) -> None:
        from naviertwin.core.uncertainty.smc import smc_sample

        rng = np.random.default_rng(0)
        # likelihood: N(2, 1)
        def loglike(x):
            return -0.5 * (x - 2.0) ** 2

        samples = smc_sample(loglike, n_particles=2000, n_steps=10, rng=rng)
        # posterior mean ≈ 2 (likelihood dominant since prior wide)
        assert abs(samples.mean() - 2.0) < 0.5

    def test_shapes(self) -> None:
        from naviertwin.core.uncertainty.smc import smc_sample

        s = smc_sample(lambda x: -0.5 * x ** 2, n_particles=100, n_steps=3)
        assert s.shape == (100,)
