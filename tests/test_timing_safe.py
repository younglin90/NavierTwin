"""Round 447 — timing-safe compare."""

from __future__ import annotations


class TestTimingSafe:
    def test_eq(self) -> None:
        from naviertwin.utils.timing_safe import equal

        assert equal("hello", "hello")
        assert not equal("hello", "world")

    def test_bytes(self) -> None:
        from naviertwin.utils.timing_safe import equal

        assert equal(b"\x01\x02", b"\x01\x02")
        assert not equal(b"\x01", b"\x02")
