"""Round 374 — Morton octree."""

from __future__ import annotations


class TestMorton:
    def test_round_trip(self) -> None:
        from naviertwin.core.amr.octree_forest import demorton3, morton3

        for x, y, z in [(0, 0, 0), (1, 2, 3), (15, 8, 4), (100, 200, 50)]:
            assert demorton3(morton3(x, y, z)) == (x, y, z)

    def test_origin_zero(self) -> None:
        from naviertwin.core.amr.octree_forest import morton3

        assert morton3(0, 0, 0) == 0

    def test_unique(self) -> None:
        from naviertwin.core.amr.octree_forest import morton3

        ids = set()
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    ids.add(morton3(x, y, z))
        assert len(ids) == 64
