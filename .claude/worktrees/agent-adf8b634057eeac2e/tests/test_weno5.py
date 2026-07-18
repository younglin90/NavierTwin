"""Round 391 — WENO5."""

from __future__ import annotations

import numpy as np


class TestWENO5:
    def test_constant_recon(self) -> None:
        from naviertwin.core.solvers.weno5 import weno5_recon_left

        u = np.full(5, 7.0)
        assert np.isclose(weno5_recon_left(u), 7.0)

    def test_linear_recon(self) -> None:
        from naviertwin.core.solvers.weno5 import weno5_recon_left

        u = np.arange(5, dtype=float)  # 0,1,2,3,4
        # linear: u(i+1/2) for i=2 with x=2 → 2.5
        v = weno5_recon_left(u)
        assert np.isclose(v, 2.5, atol=1e-10)

    def test_finite(self) -> None:
        from naviertwin.core.solvers.weno5 import weno5_recon_left

        u = np.array([1.0, 0.5, 1.0, 2.0, 1.5])
        v = weno5_recon_left(u)
        assert np.isfinite(v)
