"""Round 466 — RG coarsen."""

from __future__ import annotations

import numpy as np


class TestRG:
    def test_block_spin_majority(self) -> None:
        from naviertwin.core.multiscale.rg_coarsen import block_spin

        s = np.array([1, 1, -1, -1, -1, 1])
        out = block_spin(s, block=2)
        # blocks: (1,1)→1, (-1,-1)→-1, (-1,1)→tie → first=−1
        assert out.tolist() == [1, -1, -1]

    def test_iterate(self) -> None:
        from naviertwin.core.multiscale.rg_coarsen import rg_iterate

        s = np.array([1, 1, 1, 1, -1, -1, -1, -1])
        out = rg_iterate(s, block=2, n_iter=2)
        # 8 → 4 → 2
        assert out.shape == (2,)
