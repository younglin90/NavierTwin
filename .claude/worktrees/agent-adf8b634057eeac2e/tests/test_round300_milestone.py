"""Round 300 — D category milestone: CFD analysis (R291-R299) e2e."""

from __future__ import annotations

import numpy as np


class TestMilestoneD:
    def test_imports(self) -> None:
        from naviertwin.core.analysis import (  # noqa: F401
            delta_rortex,
            enstrophy,
            helicity,
            komega_sst,
            les_smagorinsky,
            okubo_weiss,
            rans_ke,
            vortex_core,
            ypls_correct,
        )

    def test_cavity_metrics_e2e(self) -> None:
        """Synthetic vortex 2D cavity → vorticity-based metrics consistent."""
        from naviertwin.core.analysis.enstrophy import enstrophy_density
        from naviertwin.core.analysis.helicity import helicity_density
        from naviertwin.core.analysis.okubo_weiss import okubo_weiss

        # 2D rotation gradient
        n = 5
        grad = np.zeros((n, n, 2, 2))
        for i in range(n):
            for j in range(n):
                grad[i, j] = [[0, -1], [1, 0]]
        W = okubo_weiss(grad)
        # rotation everywhere → W < 0 everywhere
        assert (W < 0).all()
        # 3D version: build vorticity vector and velocity vector
        u = np.array([[1.0, 0.0, 0.0]] * 10)
        omega = np.array([[1.0, 0.0, 0.0]] * 10)
        h = helicity_density(u, omega)
        z = enstrophy_density(omega)
        assert np.allclose(h, 1.0)
        assert np.allclose(z, 0.5)

    def test_turbulence_models(self) -> None:
        from naviertwin.core.analysis.komega_sst import eddy_viscosity_sst
        from naviertwin.core.analysis.les_smagorinsky import smagorinsky_nu_sgs
        from naviertwin.core.analysis.rans_ke import eddy_viscosity_ke

        nu_ke = eddy_viscosity_ke(np.array([1.0]), np.array([0.1]))
        nu_sst = eddy_viscosity_sst(
            np.array([1.0]), np.array([1.0]), np.array([0.0]), np.array([1.0]),
        )
        nu_les = smagorinsky_nu_sgs(np.array([1.0]), dx=0.1)
        for nu in (nu_ke, nu_sst, nu_les):
            assert np.isfinite(nu).all()
            assert (nu >= 0).all()
