"""Round 325 — bootstrap CI."""

from __future__ import annotations

import numpy as np


class TestBootstrap:
    def test_mean_ci_covers(self) -> None:
        from naviertwin.core.uncertainty.bootstrap import bootstrap_ci

        rng = np.random.default_rng(0)
        data = rng.normal(2.0, 1.0, size=200)
        lo, hi = bootstrap_ci(data, np.mean, n_boot=400, rng=rng)
        assert lo < 2.0 < hi

    def test_lo_le_hi(self) -> None:
        from naviertwin.core.uncertainty.bootstrap import bootstrap_ci

        rng = np.random.default_rng(0)
        data = rng.standard_normal(50)
        lo, hi = bootstrap_ci(data, np.std, n_boot=200, rng=rng)
        assert lo <= hi
