"""Round 419 — distillation."""

from __future__ import annotations

import numpy as np


class TestDistill:
    def test_match_zero_loss(self) -> None:
        from naviertwin.utils.distillation import distill_loss

        s = np.array([1.0, 2.0, 0.5])
        loss = distill_loss(s, s, T=2.0)
        assert abs(loss) < 1e-10

    def test_positive_when_diff(self) -> None:
        from naviertwin.utils.distillation import distill_loss

        s = np.array([1.0, 2.0])
        t = np.array([3.0, 0.0])
        loss = distill_loss(s, t, T=2.0)
        assert loss > 0
