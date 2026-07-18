"""Round 385 — level-set advect."""

from __future__ import annotations

import numpy as np


class TestLevelSetAdvect:
    def test_zero_crossing_moves_right(self) -> None:
        from naviertwin.core.geometry.levelset_advect import advect_step

        x = np.linspace(-1, 1, 41)
        phi = x.copy()
        u = np.ones_like(x)
        for _ in range(50):
            phi = advect_step(phi, u, dt=0.01, dx=2.0 / 40)
        # zero crossing has shifted toward +x
        idx = np.argmin(np.abs(phi))
        assert x[idx] > 0
