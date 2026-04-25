"""Round 531 — ONNX wrap."""

from __future__ import annotations


class TestONNXWrap:
    def test_has(self) -> None:
        from naviertwin.core.export.onnx_wrap import has_onnx, has_torch

        assert isinstance(has_onnx(), bool)
        assert isinstance(has_torch(), bool)
