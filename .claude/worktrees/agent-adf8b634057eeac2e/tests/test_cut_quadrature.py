"""Round 389 — cut-cell MC quadrature."""

from __future__ import annotations

import numpy as np


class TestQuad:
    def test_disk_area(self) -> None:
        from naviertwin.core.geometry.cut_quadrature import mc_integrate

        rng = np.random.default_rng(0)
        area = mc_integrate(
            lambda p: 1.0 if (p[0] ** 2 + p[1] ** 2) < 1.0 else 0.0,
            bbox=((-1, -1), (1, 1)), n_samples=30000, rng=rng,
        )
        assert abs(area - np.pi) < 0.1
