"""Round 480 — V category milestone: meshless (R471-R479) e2e."""

from __future__ import annotations

import numpy as np


class TestMilestoneV:
    def test_imports(self) -> None:
        from naviertwin.core.meshless import (  # noqa: F401
            dem_hertz,
            flip_pic,
            lbm_d2q9,
            mls,
            pic_deposit,
            rbf_fd,
            rbf_interp,
            sph_kernel,
            sph_momentum,
        )

    def test_lbm_couette_smoke(self) -> None:
        """LBM steps maintain mass conservation."""
        from naviertwin.core.meshless.lbm_d2q9 import equilibrium, lbm_step

        rho = np.ones((10, 10))
        u = np.zeros((2, 10, 10))
        f = equilibrium(rho, u)
        m0 = f.sum()
        for _ in range(20):
            f = lbm_step(f, omega=1.0)
        assert abs(f.sum() - m0) < 1e-9

    def test_sph_dam_break_smoke(self) -> None:
        """Density estimate from clustered particles is finite."""
        from naviertwin.core.meshless.sph_kernel import density_1d

        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(0, 1, 30))
        m = np.full(30, 0.05)
        rho = density_1d(x, m, h=0.1)
        assert np.isfinite(rho).all()
        assert (rho > 0).any()
