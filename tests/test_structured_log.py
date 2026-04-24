"""Round 238 — structured log."""

from __future__ import annotations

from pathlib import Path


class TestSLog:
    def test_emit_read(self, tmp_path: Path) -> None:
        from naviertwin.utils.structured_log import StructuredLogger

        log = StructuredLogger(tmp_path / "events.jsonl")
        log.emit("train_step", level="info", epoch=1, loss=0.3)
        log.emit("train_step", level="info", epoch=2, loss=0.2)
        log.emit("eval", level="info", r2=0.95)

        records = log.read_all()
        assert len(records) == 3
        assert records[0]["event"] == "train_step"
        assert records[0]["epoch"] == 1

    def test_filter(self, tmp_path: Path) -> None:
        from naviertwin.utils.structured_log import StructuredLogger

        log = StructuredLogger(tmp_path / "e.jsonl")
        log.emit("a", level="info", x=1)
        log.emit("b", level="warn", x=2)
        log.emit("a", level="info", x=3)

        info_a = log.filter(event="a", level="info")
        assert len(info_a) == 2

    def test_unicode(self, tmp_path: Path) -> None:
        from naviertwin.utils.structured_log import StructuredLogger

        log = StructuredLogger(tmp_path / "kor.jsonl")
        log.emit("사건", level="info", 메시지="한글 로그")
        rec = log.read_all()
        assert rec[0]["event"] == "사건"
        assert rec[0]["메시지"] == "한글 로그"
