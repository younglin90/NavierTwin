"""Round 326 — jackknife."""

from __future__ import annotations

import numpy as np


class TestJackknife:
    def test_mean_variance(self) -> None:
        from naviertwin.core.uncertainty.jackknife import jackknife_var

        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 100)
        v = jackknife_var(data, np.mean)
        # jackknife variance of mean ≈ s²/n
        s2_n = data.var(ddof=1) / len(data)
        assert abs(v - s2_n) / s2_n < 0.1

    def test_se_positive(self) -> None:
        from naviertwin.core.uncertainty.jackknife import jackknife_se

        data = np.arange(10, dtype=float)
        se = jackknife_se(data, np.mean)
        assert se > 0
