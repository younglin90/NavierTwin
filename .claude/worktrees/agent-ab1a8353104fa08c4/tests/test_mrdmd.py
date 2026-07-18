"""Round 279 — mrDMD."""

from __future__ import annotations

import numpy as np


class TestMRDMD:
    def test_tree_structure(self) -> None:
        from naviertwin.core.system_id.mrdmd import mrdmd

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 64))
        tree = mrdmd(X, max_levels=2, rank_per_level=3, slow_threshold=10.0)
        # full binary tree of depth 2: 1 + 2 + 4 = 7 nodes
        assert len(tree) == 7
        for node in tree:
            assert "level" in node
            assert "evals" in node
            assert "modes" in node

    def test_levels_within_max(self) -> None:
        from naviertwin.core.system_id.mrdmd import mrdmd

        rng = np.random.default_rng(1)
        X = rng.standard_normal((10, 32))
        tree = mrdmd(X, max_levels=3, rank_per_level=2)
        max_lvl = max(node["level"] for node in tree)
        assert max_lvl <= 3
