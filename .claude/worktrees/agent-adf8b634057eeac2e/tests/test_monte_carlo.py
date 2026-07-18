"""Round 156 — MC 적분."""

from __future__ import annotations

import numpy as np


class TestMC:
    def test_1d(self) -> None:
        from naviertwin.core.sampling.monte_carlo import mc_integral

        est, se = mc_integral(lambda x: x ** 2, 0.0, 1.0, n=50000, seed=0)
        assert abs(est - 1 / 3) < 3 * se

    def test_antithetic_lowers_variance(self) -> None:
        from naviertwin.core.sampling.monte_carlo import (
            mc_integral,
            mc_integral_antithetic,
        )

        _, se1 = mc_integral(lambda x: x, 0.0, 1.0, n=10000, seed=0)
        _, se2 = mc_integral_antithetic(lambda x: x, 0.0, 1.0, n=10000, seed=0)
        # antithetic 은 선형 f 에 대해 SE 대폭 감소 (분산 0 기대)
        assert se2 < se1 * 0.5

    def test_multivariate(self) -> None:
        from naviertwin.core.sampling.monte_carlo import mc_multivariate

        # ∫∫_{[0,1]²} x*y dx dy = 1/4
        est, se = mc_multivariate(
            lambda X: X[:, 0] * X[:, 1], [(0, 1), (0, 1)], n=30000, seed=0,
        )
        assert abs(est - 0.25) < 5 * se

    def test_importance_sampling(self) -> None:
        from naviertwin.core.sampling.monte_carlo import importance_sample

        # p = N(0,1), f(x) = x², q = N(0,2) (더 넓은 제안)
        def q_sample(n, rng):
            return rng.normal(0, np.sqrt(2), n)

        def p_pdf(x):
            return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

        def q_pdf(x):
            return np.exp(-0.25 * x ** 2) / np.sqrt(2 * 2 * np.pi)

        est, se = importance_sample(
            lambda x: x ** 2, q_sample, p_pdf, q_pdf, n=30000, seed=0,
        )
        assert abs(est - 1.0) < 5 * se  # E[x²] = 1
