"""Round 365 — level-set reinit."""

from __future__ import annotations

import numpy as np


class TestReinit:
    def test_signed_distance(self) -> None:
        from naviertwin.core.coupling.levelset_reinit import reinit_1d

        # phi initially scaled (not unit gradient)
        x = np.linspace(-2, 2, 21)
        phi = 5.0 * x  # zero level at x=0
        phi2 = reinit_1d(phi, dx=0.2, n_iter=200)
        # |phi2(x)| ≈ |x| (signed distance)
        assert np.allclose(phi2[10], 0, atol=0.5)
        # gradient close to ±1 in interior
        grad = np.diff(phi2) / 0.2
        assert (np.abs(np.abs(grad[5:-5]) - 1.0) < 0.3).all()

    def test_preserves_sign(self) -> None:
        from naviertwin.core.coupling.levelset_reinit import reinit_1d

        x = np.linspace(-1, 1, 11)
        phi = x.copy()
        phi2 = reinit_1d(phi, dx=0.2, n_iter=20)
        assert (np.sign(phi2[phi != 0]) == np.sign(phi[phi != 0])).all()
