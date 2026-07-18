"""Round 322 — ABC."""

from __future__ import annotations

import numpy as np


class TestABC:
    def test_recovers_mean(self) -> None:
        from naviertwin.core.uncertainty.abc import abc_rejection

        rng = np.random.default_rng(0)

        def sim(theta):
            return float(rng.normal(theta, 0.5))

        samples = abc_rejection(
            sim, y_obs=3.0, prior=lambda r: r.uniform(0, 6),
            eps=0.3, n_target=200, max_iter=200_000, rng=rng,
        )
        assert len(samples) >= 50  # accepted some
        # posterior centered near 3
        assert abs(samples.mean() - 3.0) < 0.3
