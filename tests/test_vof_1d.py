"""Round 364 — VOF 1D."""

from __future__ import annotations

import numpy as np


class TestVOF:
    def test_advect_right(self) -> None:
        from naviertwin.core.coupling.vof_1d import vof_step

        a = np.array([1, 1, 1, 0, 0, 0], dtype=float)
        u = np.ones(6)
        for _ in range(3):
            a = vof_step(a, u, dt=0.1, dx=0.1)
        # interface drifts right
        assert a[3] > 0

    def test_clipped(self) -> None:
        from naviertwin.core.coupling.vof_1d import vof_step

        a = np.array([0.5, 0.5, 0.5], dtype=float)
        a2 = vof_step(a, np.ones(3), dt=0.1, dx=0.1)
        assert (a2 >= 0).all() and (a2 <= 1).all()
