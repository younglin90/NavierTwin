"""Round 513 — data lineage."""

from __future__ import annotations


class TestLineage:
    def test_chain(self) -> None:
        from naviertwin.utils.data_lineage import LineageDAG

        g = LineageDAG()
        g.add("raw")
        g.add("clean", parents=["raw"])
        g.add("features", parents=["clean"])
        anc = g.ancestors("features")
        assert "raw" in anc and "clean" in anc

    def test_no_parents(self) -> None:
        from naviertwin.utils.data_lineage import LineageDAG

        g = LineageDAG()
        g.add("root")
        assert g.ancestors("root") == []
