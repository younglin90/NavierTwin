"""Round 467 — FE² skeleton."""

from __future__ import annotations

import numpy as np


class TestFE2:
    def test_linear_micro(self) -> None:
        from naviertwin.core.multiscale.fe2 import fe2_macro_stress

        s = fe2_macro_stress(np.array([0.01, 0.02]), lambda e: 100.0 * e)
        assert np.allclose(s, [1.0, 2.0])

    def test_tangent(self) -> None:
        from naviertwin.core.multiscale.fe2 import fe2_tangent_fd

        t = fe2_tangent_fd(np.array([0.01]), lambda e: 200.0 * e + 0.5 * e ** 2)
        assert abs(t[0] - 200.01) < 1e-2
