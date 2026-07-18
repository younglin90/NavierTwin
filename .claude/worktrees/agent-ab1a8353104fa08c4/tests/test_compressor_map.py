"""Round 454 — compressor map."""

from __future__ import annotations

import numpy as np


class TestCompMap:
    def test_fit_predict(self) -> None:
        from naviertwin.core.applied.compressor_map import fit_map, predict_pr

        mdot = np.linspace(1, 5, 20)
        pr = -0.1 * (mdot - 3) ** 2 + 2.7
        coef = fit_map(mdot, pr, deg=2)
        assert abs(predict_pr(coef, np.array([3.0])).item() - 2.7) < 0.05

    def test_surge_margin(self) -> None:
        from naviertwin.core.applied.compressor_map import surge_margin

        m = surge_margin(op_mdot=2.0, op_pr=2.0, surge_pr=2.5)
        assert abs(m - 0.25) < 1e-12
