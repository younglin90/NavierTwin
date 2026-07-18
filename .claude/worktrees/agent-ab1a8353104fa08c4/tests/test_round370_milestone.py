"""Round 370 — K category milestone: multiphysics imports + CHT/FSI e2e."""

from __future__ import annotations

import numpy as np


class TestMilestoneK:
    def test_imports(self) -> None:
        from naviertwin.core.coupling import (  # noqa: F401
            cahn_hilliard,
            cht,
            fsi_oneway,
            fsi_twoway,
            levelset_reinit,
            mhd_induction,
            vof_1d,
        )
        from naviertwin.core.solvers import ausm_plus, hllc_1d  # noqa: F401

    def test_cht_fsi_oneway_pipeline(self) -> None:
        from naviertwin.core.coupling.cht import cht_iterate
        from naviertwin.core.coupling.fsi_oneway import map_pressure_to_nodes

        Ts = np.linspace(300, 400, 5)
        Tf = np.linspace(500, 400, 5)
        Ts2, Tf2 = cht_iterate(Ts, Tf, k_s=8.0, k_f=2.0, n_iter=100)
        # interface continuous
        assert abs(Ts2[-1] - Tf2[0]) < 1e-9
        # FSI 1-way load on solid
        F = map_pressure_to_nodes(
            np.ones(3),
            np.tile([0., 1., 0.], (3, 1)),
            np.array([0.5, 0.5, 0.5]),
        )
        assert F.shape == (3, 3)
