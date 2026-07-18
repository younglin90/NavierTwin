"""Round 378 — ghost exchange."""

from __future__ import annotations

import numpy as np


class TestGhost:
    def test_two_blocks(self) -> None:
        from naviertwin.core.amr.ghost_exchange import exchange_ghost_1d

        a = np.array([1., 2, 3, 0, 0])
        b = np.array([0, 0, 4., 5, 6])
        blocks = [a, b]
        exchange_ghost_1d(blocks, n_ghost=2)
        assert a[-2:].tolist() == [4.0, 5.0]
        assert b[:2].tolist() == [2.0, 3.0]
