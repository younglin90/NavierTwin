"""Round 297 — swirl strength."""

from __future__ import annotations

import numpy as np


class TestSwirl:
    def test_pure_rotation(self) -> None:
        from naviertwin.core.analysis.vortex_core import swirl_strength_field

        grad = np.zeros((1, 3, 3))
        grad[0] = [[0, -1, 0], [1, 0, 0], [0, 0, 0]]  # rotation in xy
        s = swirl_strength_field(grad)
        assert np.isclose(s[0], 1.0)

    def test_pure_strain_zero(self) -> None:
        from naviertwin.core.analysis.vortex_core import swirl_strength_field

        grad = np.zeros((1, 3, 3))
        grad[0] = np.diag([1.0, -1.0, 0.0])  # pure stretching
        s = swirl_strength_field(grad)
        assert s[0] == 0.0

    def test_field_shape(self) -> None:
        from naviertwin.core.analysis.vortex_core import swirl_strength_field

        rng = np.random.default_rng(0)
        grad = rng.standard_normal((4, 5, 3, 3))
        s = swirl_strength_field(grad)
        assert s.shape == (4, 5)
