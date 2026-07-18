"""Round 519 — curriculum."""

from __future__ import annotations

import numpy as np


class TestCurr:
    def test_linear(self) -> None:
        from naviertwin.utils.curriculum import linear_difficulty

        assert linear_difficulty(epoch=0, max_epoch=10, d_min=0.1, d_max=1.0) == 0.1
        assert linear_difficulty(epoch=10, max_epoch=10, d_min=0.1, d_max=1.0) == 1.0

    def test_select(self) -> None:
        from naviertwin.utils.curriculum import select_curriculum_indices

        d = np.array([0.1, 0.5, 0.9])
        idx = select_curriculum_indices(d, threshold=0.6)
        assert idx.tolist() == [0, 1]
