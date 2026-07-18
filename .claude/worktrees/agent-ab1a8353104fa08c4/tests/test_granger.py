"""Round 424 — Granger causality."""

from __future__ import annotations

import numpy as np


class TestGranger:
    def test_y_causes_x(self) -> None:
        from naviertwin.core.analysis.granger import granger_test

        rng = np.random.default_rng(0)
        # x_t = 0.8 y_{t-1} + ε
        y = rng.standard_normal(500)
        x = np.zeros(500)
        for t in range(1, 500):
            x[t] = 0.8 * y[t - 1] + 0.05 * rng.standard_normal()
        p_dep = granger_test(x, y, lag=1)
        # independent: x' = noise
        x2 = rng.standard_normal(500)
        p_ind = granger_test(x2, y, lag=1)
        # dependent should give lower p
        assert p_dep < p_ind
