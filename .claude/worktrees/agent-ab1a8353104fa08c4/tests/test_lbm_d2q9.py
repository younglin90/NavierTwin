"""Round 479 — LBM D2Q9."""

from __future__ import annotations

import numpy as np


class TestLBM:
    def test_equilibrium_sum(self) -> None:
        from naviertwin.core.meshless.lbm_d2q9 import equilibrium

        rho = np.ones((4, 4))
        u = np.zeros((2, 4, 4))
        feq = equilibrium(rho, u)
        # Σ_k feq = rho
        assert np.allclose(feq.sum(axis=0), 1.0)

    def test_step_mass_conservation(self) -> None:
        from naviertwin.core.meshless.lbm_d2q9 import equilibrium, lbm_step

        rho = np.ones((6, 6))
        u = np.zeros((2, 6, 6))
        u[0, 3, 3] = 0.05  # small perturbation
        f = equilibrium(rho, u)
        m0 = f.sum()
        f2 = lbm_step(f, omega=1.0)
        # streaming + BGK preserves mass (periodic)
        assert abs(f2.sum() - m0) < 1e-10
