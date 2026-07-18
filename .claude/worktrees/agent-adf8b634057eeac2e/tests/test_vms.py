"""Round 464 — VMS."""

from __future__ import annotations

import numpy as np


class TestVMS:
    def test_tau_advection_dominant(self) -> None:
        from naviertwin.core.multiscale.vms import tau_supg

        # advection-dominated: τ ≈ h/(2|u|)
        t = tau_supg(u=1.0, h=0.1, nu=1e-9)
        assert abs(t - 0.05) < 1e-6

    def test_tau_diffusion_dominant(self) -> None:
        from naviertwin.core.multiscale.vms import tau_supg

        # diffusion-dominated (small u): τ ≈ h²/(4ν)
        t = tau_supg(u=1e-6, h=0.1, nu=1.0)
        assert abs(t - 0.01 / 4) < 1e-6

    def test_correction(self) -> None:
        from naviertwin.core.multiscale.vms import vms_residual_correction

        u_p = vms_residual_correction(np.array([1.0, 2.0]), tau=0.5)
        assert np.allclose(u_p, [-0.5, -1.0])
