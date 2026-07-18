"""Round 579 — coverage uplift for benettin_flow (continuous LLE)."""

from __future__ import annotations

import numpy as np


class TestBenettin:
    def test_stable_linear_negative(self) -> None:
        from naviertwin.core.analysis.lyapunov import benettin_flow

        # ẏ = -y → λ = -1 (linear, scalar)
        def rhs(t, y):
            return -y

        def jac(t, y):
            return -np.ones((1, 1))

        lam = benettin_flow(rhs, jac, y0=np.array([1.0]),
                              t_end=5.0, dt=0.05, renorm_every=10)
        # Coarse Euler on perturbation; expect negative
        assert lam < 0

    def test_unstable_linear_positive(self) -> None:
        from naviertwin.core.analysis.lyapunov import benettin_flow

        def rhs(t, y):
            return y

        def jac(t, y):
            return np.ones((1, 1))

        lam = benettin_flow(rhs, jac, y0=np.array([1e-3]),
                              t_end=2.0, dt=0.02, renorm_every=10)
        assert lam > 0
