"""Round 433 — glyph arrows."""

from __future__ import annotations

import numpy as np


class TestGlyph:
    def test_shape(self) -> None:
        from naviertwin.core.visualization.glyph import arrow_segments

        pts = np.array([[0., 0], [1., 1]])
        vec = np.array([[1., 0], [0., 1]])
        segs = arrow_segments(pts, vec, scale=2.0)
        assert segs.shape == (2, 2, 2)
        # endpoint = start + 2 * vec
        assert np.allclose(segs[0, 1], [2.0, 0.0])
        assert np.allclose(segs[1, 1], [1.0, 3.0])
