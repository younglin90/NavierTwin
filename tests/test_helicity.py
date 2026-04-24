"""Round 295 — helicity."""

from __future__ import annotations

import numpy as np


class TestHelicity:
    def test_aligned(self) -> None:
        from naviertwin.core.analysis.helicity import helicity_density

        u = np.array([[1.0, 0, 0], [0, 2.0, 0]])
        w = np.array([[3.0, 0, 0], [0, 0.5, 0]])
        h = helicity_density(u, w)
        assert np.allclose(h, [3.0, 1.0])

    def test_perpendicular_zero(self) -> None:
        from naviertwin.core.analysis.helicity import helicity_density

        u = np.array([[1.0, 0, 0]])
        w = np.array([[0.0, 1.0, 0.0]])
        assert helicity_density(u, w)[0] == 0.0

    def test_integrated(self) -> None:
        from naviertwin.core.analysis.helicity import integrated_helicity

        u = np.ones((10, 3))
        w = np.ones((10, 3))
        # 3 (per cell) * 10 cells * dV
        assert integrated_helicity(u, w, dV=2.0) == 60.0
