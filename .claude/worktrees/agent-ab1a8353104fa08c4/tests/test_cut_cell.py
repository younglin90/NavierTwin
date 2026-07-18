"""Round 382 — cut-cell."""

from __future__ import annotations

import numpy as np


class TestCutCell:
    def test_full_inside(self) -> None:
        from naviertwin.core.geometry.cut_cell import cut_cell_fraction_2d

        phi = -np.ones((3, 3))
        f = cut_cell_fraction_2d(phi)
        assert (f == 1.0).all()

    def test_full_outside(self) -> None:
        from naviertwin.core.geometry.cut_cell import cut_cell_fraction_2d

        phi = np.ones((3, 3))
        f = cut_cell_fraction_2d(phi)
        assert (f == 0.0).all()

    def test_partial(self) -> None:
        from naviertwin.core.geometry.cut_cell import cut_cell_fraction_2d

        phi = np.array([[-1, 1], [-1, 1]], dtype=float)
        f = cut_cell_fraction_2d(phi)
        assert f[0, 0] == 0.5
