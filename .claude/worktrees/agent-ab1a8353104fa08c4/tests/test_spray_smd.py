"""Round 555 — spray SMD."""

from __future__ import annotations

import numpy as np


class TestSMD:
    def test_uniform(self) -> None:
        from naviertwin.core.applied.spray_smd import sauter_mean

        d = np.full(10, 5.0)
        assert sauter_mean(d) == 5.0

    def test_mixed(self) -> None:
        from naviertwin.core.applied.spray_smd import sauter_mean

        d = np.array([1.0, 2.0])
        # (1+8)/(1+4) = 9/5 = 1.8
        assert abs(sauter_mean(d) - 1.8) < 1e-12
