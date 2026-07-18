"""Round 257 — FTLE."""

from __future__ import annotations

import numpy as np


class TestFTLE:
    def test_rotation_near_zero(self) -> None:
        """rigid rotation → strain 0 → FTLE ≈ 0."""
        from naviertwin.core.analysis.ftle import compute_ftle_2d

        def vf(p):
            return np.array([-p[1], p[0]])

        ft = compute_ftle_2d(
            vf, x=np.linspace(-0.5, 0.5, 6),
            y=np.linspace(-0.5, 0.5, 6),
            T=0.5, dt=0.01, eps=1e-3,
        )
        assert np.max(np.abs(ft)) < 0.05

    def test_saddle_positive(self) -> None:
        """saddle: u=x, v=-y → 지수 발산 → FTLE > 0."""
        from naviertwin.core.analysis.ftle import compute_ftle_2d

        def vf(p):
            return np.array([p[0], -p[1]])

        ft = compute_ftle_2d(
            vf, x=np.linspace(-0.3, 0.3, 5),
            y=np.linspace(-0.3, 0.3, 5),
            T=1.0, dt=0.01, eps=1e-3,
        )
        # FTLE ≈ max λ = 1
        assert abs(ft.mean() - 1.0) < 0.05
