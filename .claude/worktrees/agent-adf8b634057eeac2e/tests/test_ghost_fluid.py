"""Round 384 — GFM."""

from __future__ import annotations

import numpy as np


class TestGFM:
    def test_extends_value(self) -> None:
        from naviertwin.core.geometry.ghost_fluid import gfm_extend

        u = np.array([5.0, 5.0, 0.0, 0.0])
        phi = np.array([-1, -1, 1, 1], dtype=float)
        u_ext = gfm_extend(u, phi)
        # right side filled with nearest fluid-1 (value 5)
        assert u_ext[2] == 5.0
        assert u_ext[3] == 5.0

    def test_no_fluid1(self) -> None:
        from naviertwin.core.geometry.ghost_fluid import gfm_extend

        u = np.array([1.0, 2.0])
        phi = np.array([1.0, 1.0])
        u_ext = gfm_extend(u, phi)
        assert (u_ext == u).all()
