"""Round 338 — SIMP topology opt 1D."""

from __future__ import annotations

import numpy as np


class TestSIMP:
    def test_volume_constraint(self) -> None:
        from naviertwin.core.optimization.topo_simp import simp_1d

        rho = simp_1d(n=30, vol_frac=0.5, n_iter=30)
        assert rho.shape == (30,)
        # volume fraction approximately preserved
        assert abs(rho.mean() - 0.5) < 0.1

    def test_lower_bound(self) -> None:
        from naviertwin.core.optimization.topo_simp import simp_1d

        rho = simp_1d(n=20, vol_frac=0.3, n_iter=20)
        assert (rho >= 0).all() and (rho <= 1).all()
