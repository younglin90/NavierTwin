"""Round 251 — quadtree AMR."""

from __future__ import annotations

import numpy as np


class TestQTAMR:
    def test_split(self) -> None:
        from naviertwin.core.tools.quadtree_amr import QuadCell

        c = QuadCell(0, 0, 1, 1)
        c.split()
        assert len(c.children) == 4
        assert all(ch.level == 1 for ch in c.children)

    def test_refine_near_origin(self) -> None:
        from naviertwin.core.tools.quadtree_amr import (
            QuadCell,
            leaf_count,
            refine_tree,
        )

        root = QuadCell(-1, -1, 1, 1)

        def indicator(cell):
            cx, cy = cell.center
            return float(np.exp(-10 * (cx ** 2 + cy ** 2)))

        refine_tree(root, indicator, threshold=0.001, max_level=4)
        # 분할이 발생해서 level >= 1 leaf 존재
        leaves = root.leaves()
        assert leaf_count(root) >= 4
        assert any(c.level >= 1 for c in leaves)
