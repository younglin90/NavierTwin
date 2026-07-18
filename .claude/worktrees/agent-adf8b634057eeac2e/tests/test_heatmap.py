"""Round 437 — heatmap annotations."""

from __future__ import annotations

import numpy as np


class TestHeatmap:
    def test_annotations(self) -> None:
        from naviertwin.core.visualization.heatmap import annotate_heatmap

        M = np.array([[1.0, 2.5], [3.7, 4.2]])
        ann = annotate_heatmap(M, fmt="{:.1f}")
        assert ann[(0, 0)] == "1.0"
        assert ann[(1, 1)] == "4.2"
        assert len(ann) == 4
