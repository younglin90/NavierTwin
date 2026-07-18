"""Round 369 — MHD induction 1D."""

from __future__ import annotations

import numpy as np


class TestMHD:
    def test_diffusion_decays_peak(self) -> None:
        from naviertwin.core.coupling.mhd_induction import induction_step

        B = np.zeros(21)
        B[10] = 1.0
        u = np.zeros(21)  # pure diffusion
        for _ in range(20):
            B = induction_step(B, u, dt=0.001, dx=0.1, eta=0.5)
        assert B[10] < 1.0
        assert B[10] > 0
        # symmetric spread
        assert np.isclose(B[9], B[11], atol=1e-12)

    def test_finite(self) -> None:
        from naviertwin.core.coupling.mhd_induction import induction_step

        rng = np.random.default_rng(0)
        B = rng.standard_normal(15)
        u = np.ones(15)
        for _ in range(50):
            B = induction_step(B, u, dt=0.01, dx=0.1, eta=0.001)
        assert np.isfinite(B).all()
