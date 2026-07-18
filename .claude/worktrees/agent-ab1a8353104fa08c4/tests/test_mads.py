"""Round 337 — MADS."""

from __future__ import annotations

import numpy as np


class TestMADS:
    def test_quadratic(self) -> None:
        from naviertwin.core.optimization.mads import mads_minimize

        x = mads_minimize(
            lambda x: float(x @ x), x0=np.array([1.0, 2.0]),
            delta_init=0.5, delta_min=1e-4, max_iter=500,
        )
        assert np.linalg.norm(x) < 1e-2

    def test_offset_minimum(self) -> None:
        from naviertwin.core.optimization.mads import mads_minimize

        x = mads_minimize(
            lambda x: float((x[0] - 3.0) ** 2 + (x[1] + 2.0) ** 2),
            x0=np.zeros(2), delta_init=1.0, delta_min=1e-4, max_iter=500,
        )
        assert np.allclose(x, [3.0, -2.0], atol=1e-2)
