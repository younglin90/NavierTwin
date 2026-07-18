"""Round 173 — 1D 적응 격자."""

from __future__ import annotations

import numpy as np


class TestRefine:
    def test_step_refined(self) -> None:
        from naviertwin.core.tools.mesh_refine_1d import refine_by_gradient

        x = np.linspace(0, 1, 11)
        f = np.where(x < 0.5, 0.0, 1.0).astype(float)
        x2, f2 = refine_by_gradient(x, f, threshold=0.1, max_passes=3)
        assert len(x2) > len(x)
        # 정렬 유지
        assert np.all(np.diff(x2) > 0)

    def test_smooth_unchanged(self) -> None:
        from naviertwin.core.tools.mesh_refine_1d import refine_by_gradient

        x = np.linspace(0, 1, 21)
        f = np.sin(np.pi * x)  # smooth
        x2, _ = refine_by_gradient(x, f, threshold=0.5)
        assert len(x2) == len(x)

    def test_coarsen(self) -> None:
        from naviertwin.core.tools.mesh_refine_1d import coarsen_by_tolerance

        x = np.linspace(0, 1, 101)
        f = np.linspace(0, 1, 101)  # linear → 2nd diff = 0
        xc, fc = coarsen_by_tolerance(x, f, tol=1e-6)
        # 끝점만 남아야 함
        assert len(xc) == 2
        assert fc[0] == 0.0 and fc[-1] == 1.0
