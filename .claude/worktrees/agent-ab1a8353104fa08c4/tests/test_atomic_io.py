"""Round 94 — 원자적 파일 쓰기."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestAtomicIO:
    def test_text(self, tmp_path: Path) -> None:
        from naviertwin.utils.atomic_io import atomic_write_text

        p = tmp_path / "sub" / "x.txt"
        atomic_write_text(p, "hello 한글")
        assert p.read_text(encoding="utf-8") == "hello 한글"

    def test_bytes(self, tmp_path: Path) -> None:
        from naviertwin.utils.atomic_io import atomic_write_bytes

        p = tmp_path / "x.bin"
        atomic_write_bytes(p, b"\x00\x01\x02")
        assert p.read_bytes() == b"\x00\x01\x02"

    def test_open_context(self, tmp_path: Path) -> None:
        from naviertwin.utils.atomic_io import atomic_open

        p = tmp_path / "x.txt"
        with atomic_open(p, "w") as f:
            f.write("abc")
        assert p.read_text() == "abc"

    def test_failure_preserves_old(self, tmp_path: Path) -> None:
        from naviertwin.utils.atomic_io import atomic_open

        p = tmp_path / "x.txt"
        p.write_text("original")
        with pytest.raises(ValueError):
            with atomic_open(p, "w") as f:
                f.write("partial")
                raise ValueError("boom")
        # 원본 유지
        assert p.read_text() == "original"
        # 임시 파일은 정리됨
        leftovers = [x for x in tmp_path.iterdir() if x.name.startswith(".")]
        assert leftovers == []
