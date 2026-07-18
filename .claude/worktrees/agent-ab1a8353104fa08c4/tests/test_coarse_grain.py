"""Round 465 — coarse grain."""

from __future__ import annotations

import numpy as np


class TestCG:
    def test_block(self) -> None:
        from naviertwin.core.multiscale.coarse_grain import block_average

        out = block_average(np.arange(10.0), block=5)
        assert np.allclose(out, [2.0, 7.0])

    def test_moving(self) -> None:
        from naviertwin.core.multiscale.coarse_grain import moving_average

        out = moving_average(np.ones(5), window=3)
        assert np.allclose(out, 1.0)
        assert out.shape == (3,)
