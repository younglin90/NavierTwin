"""Round 363 — FSI 2-way."""

from __future__ import annotations

import numpy as np


class TestFSI2way:
    def test_converges_linear(self) -> None:
        from naviertwin.core.coupling.fsi_twoway import fsi_2way

        # fluid: load = 2 - 0.5 x; solid: x = load / 1.0
        # fixed point: x = 2 - 0.5 x → x = 4/3
        x = fsi_2way(
            fluid_solve=lambda x: 2.0 - 0.5 * x,
            solid_solve=lambda load: load,
            x0=np.array([0.0]),
            n_iter=50,
        )
        assert abs(x[0] - 4.0 / 3.0) < 1e-4

    def test_aitken_returns_finite(self) -> None:
        from naviertwin.core.coupling.fsi_twoway import aitken_relax

        w = aitken_relax(0.5, np.array([1.0]), np.array([0.6]))
        assert np.isfinite(w)
