"""Round 431 — volume render."""

from __future__ import annotations

import numpy as np


class TestVolumeRender:
    def test_blob_visible(self) -> None:
        from naviertwin.core.visualization.volume_render import ray_march

        vol = np.zeros((30, 30, 30))
        vol[10:20, 10:20, 10:20] = 1.0
        img = ray_march(vol, n_steps=30, axis=2)
        # blob region brighter
        assert img[15, 15] > img[0, 0]

    def test_zero_volume(self) -> None:
        from naviertwin.core.visualization.volume_render import ray_march

        img = ray_march(np.zeros((10, 10, 10)))
        assert img.shape == (10, 10)
        assert (img == 0).all()
