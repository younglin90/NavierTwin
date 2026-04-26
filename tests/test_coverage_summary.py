"""Round 566 — coverage parser."""

from __future__ import annotations


class TestCoverage:
    def test_parse(self) -> None:
        from naviertwin.utils.coverage_summary import parse_coverage_text

        text = (
            "Name                  Stmts   Miss  Cover\n"
            "----------------------------------------\n"
            "src/naviertwin/a.py     100     10    90%\n"
            "src/naviertwin/b.py      50     30    40%\n"
            "TOTAL                   150     40    73%\n"
        )
        rows = parse_coverage_text(text)
        names = {r["name"] for r in rows}
        assert "src/naviertwin/a.py" in names
        assert "TOTAL" in names

    def test_below_threshold(self) -> None:
        from naviertwin.utils.coverage_summary import (
            below_threshold,
            parse_coverage_text,
        )

        text = (
            "src/x.py    10   2   80%\n"
            "src/y.py    10   8   20%\n"
            "TOTAL       20  10   50%\n"
        )
        rows = parse_coverage_text(text)
        weak = below_threshold(rows, min_pct=70)
        assert len(weak) == 1
        assert weak[0]["name"] == "src/y.py"
