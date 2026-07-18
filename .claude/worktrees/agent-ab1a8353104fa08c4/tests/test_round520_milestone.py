"""Round 520 — Z category milestone: dataset/lineage (R511-R519) e2e."""

from __future__ import annotations


class TestMilestoneZ:
    def test_imports(self) -> None:
        from naviertwin.utils import (  # noqa: F401
            active_learning,
            class_weights,
            curriculum,
            data_lineage,
            dataset_card,
            dataset_cas,
            dataset_metadata,
            dataset_split,
            pseudo_label,
        )

    def test_card_lineage_e2e(self, tmp_path) -> None:
        from naviertwin.utils.data_lineage import LineageDAG
        from naviertwin.utils.dataset_card import write_card
        from naviertwin.utils.dataset_cas import cas_hash, version_id
        from naviertwin.utils.dataset_metadata import write_metadata

        h = cas_hash(b"snapshot data")
        vid = version_id("snap", h)
        write_card(tmp_path / "card.md", name=vid, n_samples=100, source="DNS")
        write_metadata(tmp_path / "m.json", {
            "name": vid, "version": "1.0", "n_samples": 100,
        })
        g = LineageDAG()
        g.add("raw")
        g.add("snap", parents=["raw"], meta={"hash": h})
        assert "raw" in g.ancestors("snap")
        assert (tmp_path / "card.md").exists()
