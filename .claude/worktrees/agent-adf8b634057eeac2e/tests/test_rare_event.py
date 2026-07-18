"""Round 327 — rare-event probability."""

from __future__ import annotations

import numpy as np


class TestRareEvent:
    def test_returns_finite_probability(self) -> None:
        from naviertwin.core.uncertainty.rare_event import subset_simulation

        rng = np.random.default_rng(0)
        # P(X > 3) for standard normal ≈ 1.35e-3
        p = subset_simulation(
            g=lambda x: x[:, 0],
            d=1, b=3.0, p0=0.1, n=500, rng=rng,
        )
        assert 0 <= p <= 1
        assert np.isfinite(p)

    def test_lower_threshold_higher_p(self) -> None:
        from naviertwin.core.uncertainty.rare_event import subset_simulation

        rng = np.random.default_rng(0)
        p_easy = subset_simulation(
            g=lambda x: x[:, 0], d=1, b=1.0, p0=0.1, n=500, rng=rng,
        )
        p_hard = subset_simulation(
            g=lambda x: x[:, 0], d=1, b=3.0, p0=0.1, n=500, rng=rng,
        )
        assert p_easy >= p_hard
