"""Round 399 — entropy-stable KEP flux."""

from __future__ import annotations

import numpy as np


class TestES:
    def test_consistency(self) -> None:
        """UL=UR → F = physical flux."""
        from naviertwin.core.solvers.entropy_stable import kep_flux

        U = np.array([1.0, 0.5, 2.0])
        F = kep_flux(U, U, gamma=1.4)
        # ρu = 0.5
        assert np.isclose(F[0], 0.5)

    def test_shape(self) -> None:
        from naviertwin.core.solvers.entropy_stable import kep_flux

        F = kep_flux(np.array([1.0, 0.0, 2.5]), np.array([0.5, 0.1, 1.0]))
        assert F.shape == (3,)
        assert np.isfinite(F).all()
