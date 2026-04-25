"""Round 463 — projective integration."""

from __future__ import annotations

import numpy as np


class TestProjective:
    def test_decay(self) -> None:
        from naviertwin.core.multiscale.projective_integ import projective_step

        def micro(u, dt):
            return u * np.exp(-dt)

        u = np.array([1.0])
        for _ in range(5):
            u = projective_step(u, micro, dt_micro=0.01, n_micro=3, dt_big=0.5)
        # roughly e^{-1} - e^{-3}
        assert np.isfinite(u).all()
        assert u[0] < 1.0
