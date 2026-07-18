"""Round 524 — checkpoint manager."""

from __future__ import annotations


class TestCkpt:
    def test_keep_top_n(self, tmp_path) -> None:
        from naviertwin.utils.workflow.checkpoint_mgr import CheckpointManager

        m = CheckpointManager(tmp_path, keep=2, mode="max")
        for s in [0.1, 0.5, 0.3, 0.9, 0.2]:
            m.add(score=s, payload=b"x")
        # only top 2 (0.9, 0.5) remain
        assert len(m.index) == 2
        scores = sorted([r["score"] for r in m.index])
        assert scores == [0.5, 0.9]
