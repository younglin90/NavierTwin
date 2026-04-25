"""Round 532 — TorchScript wrap."""

from __future__ import annotations

import pytest


class TestTSWrap:
    def test_trace(self, tmp_path) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.export.torchscript_wrap import trace_module

        m = torch.nn.Linear(3, 2)
        x = torch.randn(1, 3)
        out_path = tmp_path / "m.pt"
        traced = trace_module(m, x, save_path=out_path)
        assert out_path.exists()
        # traced model executes
        y = traced(x)
        assert y.shape == (1, 2)
