"""Round 523 — sweep."""

from __future__ import annotations


class TestSweep:
    def test_grid(self) -> None:
        from naviertwin.utils.workflow.sweep import grid_sweep

        out = list(grid_sweep({"a": [1, 2], "b": [3, 4]}))
        assert len(out) == 4

    def test_random(self) -> None:
        from naviertwin.utils.workflow.sweep import random_sweep

        out = random_sweep({"lr": (0.001, 0.1)}, n=5, seed=0)
        assert len(out) == 5
        for p in out:
            assert 0.001 <= p["lr"] <= 0.1
