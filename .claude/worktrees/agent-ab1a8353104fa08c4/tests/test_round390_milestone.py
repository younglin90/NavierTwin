"""Round 390 — M category milestone: geometry (R381-R389) cut-cell e2e."""

from __future__ import annotations

import numpy as np


class TestMilestoneM:
    def test_imports(self) -> None:
        from naviertwin.core.geometry import (  # noqa: F401
            csg,
            cut_cell,
            cut_quadrature,
            ghost_fluid,
            ib_forcing,
            levelset_advect,
            remap_1d,
            sdf,
            sdf_reinit,
        )

    def test_sdf_cut_cell_pipeline(self) -> None:
        """SDF (sphere) → cut-cell fraction grid."""
        from naviertwin.core.geometry.cut_cell import cut_cell_fraction_2d
        from naviertwin.core.geometry.sdf import sdf_sphere

        # 20x20 grid in [-1, 1]²
        n = 20
        x = np.linspace(-1, 1, n + 1)
        X, Y = np.meshgrid(x, x, indexing="ij")
        pts = np.stack([X, Y, np.zeros_like(X)], axis=-1)
        # phi at corners
        phi = np.linalg.norm(pts, axis=-1) - 0.5
        frac = cut_cell_fraction_2d(phi)
        assert frac.shape == (n, n)
        # interior cells fully inside circle (frac=1)
        # use center of disk (small radius to ensure full inside)
        # check fraction is between 0 and 1
        assert (frac >= 0).all() and (frac <= 1).all()
        # area sum × dx² ≈ π * 0.25 ≈ 0.785
        dx = 2.0 / n
        area = frac.sum() * dx * dx
        assert abs(area - np.pi * 0.25) < 0.1

        # also check sdf_sphere works
        d = sdf_sphere(np.zeros(3), center=np.zeros(3), r=0.5)
        assert d == -0.5
