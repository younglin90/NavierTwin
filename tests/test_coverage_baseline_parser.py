"""Round 576 — coverage-baseline parser unit test."""

from __future__ import annotations


class TestCovParser:
    def test_parse_total_and_modules(self) -> None:
        from scripts.capture_coverage_baseline import (
            parse,  # type: ignore[import-not-found]  # noqa: E501
        )

        text = (
            "Name                              Stmts   Miss  Cover\n"
            "src/naviertwin/a.py                  10      2    80%\n"
            "src/naviertwin/sub/b.py             100     50    50%\n"
            "TOTAL                               150     52    65%\n"
        )
        total, mods = parse(text)
        assert total == 65
        names = {m["name"] for m in mods}
        assert "src/naviertwin/a.py" in names
        assert "src/naviertwin/sub/b.py" in names
        # both 80, 50
        covers = sorted(m["cover"] for m in mods)
        assert covers == [50, 80]

    def test_parse_with_missing_lines_column(self) -> None:
        from scripts.capture_coverage_baseline import parse  # type: ignore[import-not-found]

        # pytest-cov adds trailing missing-line numbers
        text = "src/x.py    50   5   90%   12-15, 30\nTOTAL  50   5   90%\n"
        total, mods = parse(text)
        assert total == 90
        assert mods[0]["name"] == "src/x.py"
        assert mods[0]["cover"] == 90
