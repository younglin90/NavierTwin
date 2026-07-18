"""Round 80 — Welford rolling stats."""

from __future__ import annotations

import numpy as np


class TestWelford:
    def test_scalar_matches_numpy(self) -> None:
        from naviertwin.utils.rolling_stats import WelfordStats

        rng = np.random.default_rng(0)
        data = rng.standard_normal(500)
        s = WelfordStats()
        for v in data:
            s.update(float(v))
        assert abs(s.mean - data.mean()) < 1e-10
        assert abs(s.variance - data.var(ddof=1)) < 1e-10

    def test_vector(self) -> None:
        from naviertwin.utils.rolling_stats import WelfordStats

        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 5))
        s = WelfordStats()
        for row in X:
            s.update(row)
        assert np.allclose(s.mean, X.mean(axis=0), atol=1e-10)
        assert np.allclose(s.variance, X.var(axis=0, ddof=1), atol=1e-10)

    def test_merge(self) -> None:
        from naviertwin.utils.rolling_stats import WelfordStats

        rng = np.random.default_rng(2)
        data = rng.standard_normal(300)
        a, b = WelfordStats(), WelfordStats()
        for v in data[:150]:
            a.update(float(v))
        for v in data[150:]:
            b.update(float(v))
        a.merge(b)
        assert abs(a.mean - data.mean()) < 1e-10
        assert abs(a.variance - data.var(ddof=1)) < 1e-10
