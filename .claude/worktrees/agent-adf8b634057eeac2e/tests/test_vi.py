"""Round 212 — VI."""

from __future__ import annotations

import numpy as np


class TestVI:
    def test_gaussian_posterior(self) -> None:
        from naviertwin.core.uncertainty.vi import mean_field_vi

        def logp(z):
            return -0.5 * float(((z[0] - 3.0) / 0.4) ** 2)

        mu, log_sigma = mean_field_vi(
            logp, dim=1, n_iter=100, lr=0.02, mc_samples=4, seed=0,
        )
        # 0에서 시작해 ELBO ascent → μ > 0 (target 3 방향)
        assert float(mu[0]) > 0.0
        assert np.isfinite(float(log_sigma[0]))
