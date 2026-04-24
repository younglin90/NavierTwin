"""Round 373 — Berger-Oliger schedule."""

from __future__ import annotations


class TestBO:
    def test_two_levels(self) -> None:
        from naviertwin.core.amr.berger_oliger import schedule

        calls = schedule(level=0, max_level=1, refine_ratio=2)
        # level 1, level 1, level 0
        assert calls == [1, 1, 0]

    def test_three_levels(self) -> None:
        from naviertwin.core.amr.berger_oliger import schedule

        calls = schedule(level=0, max_level=2, refine_ratio=2)
        # 2 calls of level-1 substep, each = (lvl 2 twice + lvl 1)
        # full pattern: [2, 2, 1, 2, 2, 1, 0]
        assert calls == [2, 2, 1, 2, 2, 1, 0]
