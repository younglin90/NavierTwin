"""Round 531 + 597 — ONNX wrap full coverage."""

from __future__ import annotations

import builtins

import pytest


class TestONNXWrap:
    def test_has(self) -> None:
        from naviertwin.core.export.onnx_wrap import has_onnx, has_torch

        assert isinstance(has_onnx(), bool)
        assert isinstance(has_torch(), bool)

    def test_has_torch_false_when_blocked(self, monkeypatch) -> None:
        from naviertwin.core.export import onnx_wrap

        real_import = builtins.__import__

        def block(name, *a, **kw):
            if name == "torch":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block)
        assert onnx_wrap.has_torch() is False

    def test_has_onnx_false_when_blocked(self, monkeypatch) -> None:
        from naviertwin.core.export import onnx_wrap

        real_import = builtins.__import__

        def block(name, *a, **kw):
            if name == "onnx":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block)
        assert onnx_wrap.has_onnx() is False

    def test_safe_export_no_torch(self, monkeypatch, tmp_path) -> None:
        from naviertwin.core.export import onnx_wrap

        real_import = builtins.__import__

        def block(name, *a, **kw):
            if name == "torch":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block)
        result = onnx_wrap.safe_export(None, None, tmp_path / "out.onnx")
        assert result is False

    def test_safe_export_torch_exception(self, monkeypatch, tmp_path) -> None:
        torch = pytest.importorskip("torch")
        import torch.nn as nn

        from naviertwin.core.export.onnx_wrap import safe_export

        class BrokenModel(nn.Module):
            def forward(self, x):
                raise RuntimeError("broken")

        path = tmp_path / "broken.onnx"
        result = safe_export(
            BrokenModel(),
            torch.zeros(1, 4),
            path,
        )
        assert result is False
