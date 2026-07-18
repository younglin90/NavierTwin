"""Round 418 — magnitude pruning."""

from __future__ import annotations

import numpy as np


class TestPruning:
    def test_50_percent(self) -> None:
        from naviertwin.utils.pruning import prune_magnitude

        w = np.array([1.0, -0.1, 2.0, 0.05])
        wp = prune_magnitude(w, sparsity=0.5)
        assert (wp == 0).sum() == 2
        # large values preserved
        assert wp[0] == 1.0
        assert wp[2] == 2.0

    def test_zero_sparsity(self) -> None:
        from naviertwin.utils.pruning import prune_magnitude

        w = np.arange(10.0)
        wp = prune_magnitude(w, sparsity=0.0)
        assert np.allclose(wp, w)
