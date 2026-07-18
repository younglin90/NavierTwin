"""Round 412 — mixed precision."""

from __future__ import annotations


class TestAMP:
    def test_context_no_torch_or_cpu(self) -> None:
        from naviertwin.utils.mixed_precision import amp_context

        with amp_context(dtype="float16", device="cpu"):
            x = 1 + 1
        assert x == 2

    def test_default(self) -> None:
        from naviertwin.utils.mixed_precision import amp_context

        with amp_context():
            pass
