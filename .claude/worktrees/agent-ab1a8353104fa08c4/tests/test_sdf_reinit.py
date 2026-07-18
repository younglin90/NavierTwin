"""Round 386 — SDF reinit."""

from __future__ import annotations

import numpy as np


class TestSDFReinit:
    def test_unit_gradient(self) -> None:
        from naviertwin.core.geometry.sdf_reinit import fast_march_1d

        phi = np.array([5.0, 3.0, 0.0, -3.0, -5.0])
        phi2 = fast_march_1d(phi, dx=1.0)
        # at zero
        assert phi2[2] == 0.0
        # |gradient| = 1
        assert np.allclose(np.abs(np.diff(phi2)), 1.0)

    def test_no_interface(self) -> None:
        from naviertwin.core.geometry.sdf_reinit import fast_march_1d

        phi = np.array([1.0, 2.0, 3.0])
        phi2 = fast_march_1d(phi, dx=1.0)
        assert (phi2 == phi).all()
