"""Round 518 — pseudo labeling."""

from __future__ import annotations

import numpy as np


class TestPL:
    def test_filter(self) -> None:
        from naviertwin.utils.pseudo_label import pseudo_label_filter

        probs = np.array([[0.95, 0.05], [0.55, 0.45], [0.99, 0.01]])
        idx, lbl = pseudo_label_filter(probs, threshold=0.9)
        assert idx.tolist() == [0, 2]
        assert lbl.tolist() == [0, 0]

    def test_consistency(self) -> None:
        from naviertwin.utils.pseudo_label import consistency_filter

        a = np.array([[0.95, 0.05], [0.95, 0.05]])
        b = np.array([[0.92, 0.08], [0.4, 0.6]])
        idx = consistency_filter(a, b, threshold=0.9)
        # only idx 0: both confident and same
        assert idx.tolist() == [0]
