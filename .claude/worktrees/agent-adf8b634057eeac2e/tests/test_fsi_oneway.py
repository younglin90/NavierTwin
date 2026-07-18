"""Round 362 — FSI 1-way pressure mapper."""

from __future__ import annotations

import numpy as np


class TestFSI1way:
    def test_force_dir_opposite_normal(self) -> None:
        from naviertwin.core.coupling.fsi_oneway import map_pressure_to_nodes

        p = np.array([2.0])
        n = np.array([[0., 1.0, 0.]])
        a = np.array([3.0])
        F = map_pressure_to_nodes(p, n, a)
        assert F.shape == (1, 3)
        # F = -p A n → -6 in y
        assert np.allclose(F[0], [0, -6.0, 0])

    def test_zero_pressure(self) -> None:
        from naviertwin.core.coupling.fsi_oneway import map_pressure_to_nodes

        F = map_pressure_to_nodes(
            np.zeros(3), np.tile([1., 0, 0], (3, 1)), np.ones(3),
        )
        assert np.allclose(F, 0.0)
