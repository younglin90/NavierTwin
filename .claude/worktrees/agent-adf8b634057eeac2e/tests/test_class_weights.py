"""Round 516 — class weights."""

from __future__ import annotations

import numpy as np


class TestCW:
    def test_balanced(self) -> None:
        from naviertwin.utils.class_weights import balanced_weights

        # 75% class 0, 25% class 1 → weights inversely
        w = balanced_weights(np.array([0, 0, 0, 1]))
        # n=4, classes=2 → w[0] = 4/(2*3) = 0.667, w[1] = 4/(2*1)=2.0
        assert abs(w[0] - 4 / 6) < 1e-9
        assert abs(w[1] - 2.0) < 1e-9

    def test_per_sample(self) -> None:
        from naviertwin.utils.class_weights import per_sample_weights

        ws = per_sample_weights(np.array([0, 1]))
        # equal classes → both 1.0
        assert np.allclose(ws, 1.0)
