"""Round 305 — streamline seeds."""

from __future__ import annotations

import numpy as np


class TestSeeds:
    def test_uniform_in_bbox(self) -> None:
        from naviertwin.core.tools.stream_seeds import uniform_seeds

        seeds = uniform_seeds(bbox=((0, 0), (1, 2)), n=20)
        assert seeds.shape == (20, 2)
        assert ((seeds[:, 0] >= 0) & (seeds[:, 0] <= 1)).all()
        assert ((seeds[:, 1] >= 0) & (seeds[:, 1] <= 2)).all()

    def test_vorticity_concentrates_on_high_w(self) -> None:
        from naviertwin.core.tools.stream_seeds import vorticity_weighted_seeds

        w = np.zeros((10, 10))
        w[7, 7] = 100.0  # single hot spot
        seeds = vorticity_weighted_seeds(
            bbox=((0, 0), (1, 1)), vorticity_field=w, n=200, seed=0,
        )
        # most seeds near (0.75, 0.75)
        center = np.array([0.75, 0.75])
        d = np.linalg.norm(seeds - center, axis=1)
        assert (d < 0.15).mean() > 0.9
