"""Round 547 — what-if."""

from __future__ import annotations


class TestWhatIf:
    def test_override(self) -> None:
        from naviertwin.core.twin.whatif import what_if

        base = {"lr": 0.01, "bs": 32}
        out = what_if(base, {"lr": 0.05}, lambda cfg: cfg["lr"])
        assert out == 0.05
        # base untouched
        assert base["lr"] == 0.01
