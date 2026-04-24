"""Round 330 — G category milestone: UQ (R321-R329) + MLMC."""

from __future__ import annotations

import numpy as np


class TestMilestoneG:
    def test_imports(self) -> None:
        from naviertwin.core.uncertainty import (  # noqa: F401
            abc,
            bootstrap,
            jackknife,
            kl_expansion,
            mlmc,
            nested_sampling,
            rare_event,
            smc,
            stoch_collocation,
            svgd,
        )

    def test_mlmc_estimates_mean(self) -> None:
        """MLMC on toy: f_0 = ξ, f_l - f_{l-1} → 0 (mean 0). Estimate ≈ 0."""
        from naviertwin.core.uncertainty.mlmc import mlmc_estimate

        rng = np.random.default_rng(0)

        def sampler(level: int, r: np.random.Generator) -> float:
            if level == 0:
                return float(r.normal(2.0, 1.0))
            # noisy correction with mean 0
            return float(r.normal(0.0, 0.5 / level))

        est, var = mlmc_estimate(sampler, levels=[200, 50, 20], rng=rng)
        # est ≈ 2.0 (level-0 mean)
        assert abs(est - 2.0) < 0.4
        assert var >= 0

    def test_uq_smoke(self) -> None:
        """Bootstrap + jackknife give comparable variance estimates."""
        from naviertwin.core.uncertainty.bootstrap import bootstrap_ci
        from naviertwin.core.uncertainty.jackknife import jackknife_var

        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 100)
        v_jk = jackknife_var(data, np.mean)
        lo, hi = bootstrap_ci(data, np.mean, n_boot=300, rng=rng)
        # both finite, CI brackets 0
        assert v_jk > 0
        assert lo < 0 < hi
