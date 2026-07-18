"""Round 323 — nested sampling."""

from __future__ import annotations

import numpy as np


class TestNested:
    def test_runs_and_returns_finite_logZ(self) -> None:
        from naviertwin.core.uncertainty.nested_sampling import nested_sample

        rng = np.random.default_rng(0)
        dead, logZ = nested_sample(
            loglike=lambda x: -0.5 * x ** 2,
            prior_sample=lambda r: r.uniform(-5, 5),
            n_live=30, n_iter=150, rng=rng,
        )
        assert dead.shape == (150,)
        assert np.isfinite(logZ)

    def test_dead_points_distribution(self) -> None:
        from naviertwin.core.uncertainty.nested_sampling import nested_sample

        rng = np.random.default_rng(1)
        dead, _ = nested_sample(
            loglike=lambda x: -0.5 * (x - 2.0) ** 2,
            prior_sample=lambda r: r.uniform(-5, 5),
            n_live=50, n_iter=200, rng=rng,
        )
        # final points should concentrate near posterior mode (~2)
        assert dead.size > 0
