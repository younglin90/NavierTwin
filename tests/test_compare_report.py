"""Round 529 — comparison report."""

from __future__ import annotations


class TestCompare:
    def test_md(self) -> None:
        from naviertwin.utils.workflow.compare_report import compare_md

        out = compare_md(
            [{"name": "A", "acc": 0.9}, {"name": "B", "acc": 0.85}],
            columns=["name", "acc"],
        )
        assert "| name | acc |" in out
        assert "| A | 0.9 |" in out
        assert "| B | 0.85 |" in out
