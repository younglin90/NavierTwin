"""Round 491 — spectral norm."""

from __future__ import annotations

import numpy as np


class TestSN:
    def test_diag(self) -> None:
        from naviertwin.utils.spectral_norm import spectral_norm

        W = np.diag([3.0, 1.0, 0.5])
        assert abs(spectral_norm(W) - 3.0) < 1e-3

    def test_penalty(self) -> None:
        from naviertwin.utils.spectral_norm import sn_penalty

        W = np.diag([2.0, 1.0])
        # spectral norm 2, target 1 → penalty = (2-1)^2 = 1
        assert abs(sn_penalty(W, target=1.0) - 1.0) < 1e-2

    def test_below_target(self) -> None:
        from naviertwin.utils.spectral_norm import sn_penalty

        W = np.diag([0.5])
        assert sn_penalty(W, target=1.0) == 0.0
