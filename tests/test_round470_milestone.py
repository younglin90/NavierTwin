"""Round 470 — U category milestone: multiscale (R461-R469) HMM e2e."""

from __future__ import annotations

import numpy as np


class TestMilestoneU:
    def test_imports(self) -> None:
        from naviertwin.core.multiscale import (  # noqa: F401
            coarse_grain,
            fe2,
            gap_tooth,
            hmm,
            homogenization,
            periodic_stress,
            projective_integ,
            rg_coarsen,
            vms,
        )

    def test_hmm_macro_micro_e2e(self) -> None:
        """Macro state → micro homogenized k → effective conductivity."""
        from naviertwin.core.multiscale.hmm import hmm_macro_flux
        from naviertwin.core.multiscale.homogenization import effective_conductivity_1d

        # micro at each macro point: cell with two-phase coefficients
        def micro(u_macro: float) -> float:
            cell = np.array([1.0, 4.0])
            return effective_conductivity_1d(cell) * u_macro

        out = hmm_macro_flux(np.array([1.0, 2.0]), micro)
        # k_eff = 1.6 → flux = 1.6, 3.2
        assert np.allclose(out, [1.6, 3.2])
