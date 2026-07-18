"""Round 366 — Cahn-Hilliard 1D."""

from __future__ import annotations

import numpy as np


class TestCH:
    def test_mass_conservation(self) -> None:
        from naviertwin.core.coupling.cahn_hilliard import ch_step

        rng = np.random.default_rng(0)
        c = 0.5 + 0.05 * rng.standard_normal(40)
        m0 = c.sum()
        for _ in range(50):
            c = ch_step(c, dt=1e-5, dx=0.1, M=1.0, eps=0.05)
        # mass conserved (periodic, ∇² → divergence)
        assert abs(c.sum() - m0) < 1e-8
        assert np.isfinite(c).all()
