"""Round 383 — IB forcing."""

from __future__ import annotations

import numpy as np


class TestIB:
    def test_force_pushes_to_target(self) -> None:
        from naviertwin.core.geometry.ib_forcing import direct_forcing

        u = np.array([1.0, 1.0])
        f = direct_forcing(u, np.zeros(2), np.array([True, False]), dt=0.1)
        assert f[0] == -10.0
        assert f[1] == 0.0

    def test_no_mask_no_force(self) -> None:
        from naviertwin.core.geometry.ib_forcing import direct_forcing

        u = np.ones(3)
        f = direct_forcing(u, np.zeros(3), np.zeros(3, dtype=bool), dt=0.5)
        assert (f == 0).all()
