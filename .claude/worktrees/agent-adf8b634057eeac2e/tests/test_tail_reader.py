"""Round 168 — TailReader."""

from __future__ import annotations

import time
from pathlib import Path


class TestTail:
    def test_appends(self, tmp_path: Path) -> None:
        from naviertwin.core.streaming.tail_reader import TailReader

        p = tmp_path / "log.txt"
        p.write_text("old line\n", encoding="utf-8")

        lines: list[str] = []
        with TailReader(p, lines.append, poll_interval=0.02):
            with p.open("a", encoding="utf-8") as f:
                f.write("new 1\n")
                f.write("new 2\n")
                f.flush()
            for _ in range(30):
                if len(lines) >= 2:
                    break
                time.sleep(0.02)

        assert "new 1" in lines
        assert "new 2" in lines
        assert "old line" not in lines
