"""Round 296 — enstrophy."""

from __future__ import annotations

import numpy as np


class TestEnstrophy:
    def test_density_formula(self) -> None:
        from naviertwin.core.analysis.enstrophy import enstrophy_density

        w = np.array([[3.0, 4.0, 0.0]])
        z = enstrophy_density(w)
        assert z[0] == 12.5  # 0.5 * 25

    def test_zero_field(self) -> None:
        from naviertwin.core.analysis.enstrophy import enstrophy_density

        w = np.zeros((5, 3))
        assert (enstrophy_density(w) == 0).all()

    def test_integrated(self) -> None:
        from naviertwin.core.analysis.enstrophy import integrated_enstrophy

        w = np.ones((4, 3))
        # |ω|²/2 = 1.5 per cell; 4 cells; dV=2 → 12.0
        assert integrated_enstrophy(w, dV=2.0) == 12.0
