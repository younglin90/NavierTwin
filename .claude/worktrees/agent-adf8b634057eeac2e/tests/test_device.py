"""Round 149 + 598 — device 유틸 (full coverage)."""

from __future__ import annotations

import builtins

import pytest


class TestDevice:
    def test_auto(self) -> None:
        pytest.importorskip("torch")
        from naviertwin.utils.device import get_device

        d = get_device("auto")
        assert str(d) in ("cpu", "cuda", "mps", "cuda:0")

    def test_cpu_force(self) -> None:
        pytest.importorskip("torch")
        from naviertwin.utils.device import get_device

        assert str(get_device("cpu")) == "cpu"

    def test_move_nested(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.utils.device import get_device, move_to

        d = get_device("cpu")
        obj = {"x": torch.zeros(3), "y": [torch.ones(2), (torch.zeros(1),)]}
        out = move_to(obj, d)
        assert out["x"].device.type == "cpu"

    def test_memory_info(self) -> None:
        from naviertwin.utils.device import available_memory_mb

        info = available_memory_mb()
        assert "cuda" in info

    def test_cuda_unavailable_raises(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.utils.device import get_device

        if torch.cuda.is_available():
            pytest.skip("CUDA present — can't test this path")
        with pytest.raises(RuntimeError, match="CUDA"):
            get_device("cuda")

    def test_get_device_no_torch(self, monkeypatch) -> None:
        from naviertwin.utils import device as dev_mod

        real_import = builtins.__import__

        def block(name, *a, **kw):
            if name == "torch":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block)
        with pytest.raises(RuntimeError, match="torch"):
            dev_mod.get_device("cpu")

    def test_move_to_no_torch(self, monkeypatch) -> None:
        from naviertwin.utils import device as dev_mod

        real_import = builtins.__import__

        def block(name, *a, **kw):
            if name == "torch":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block)
        with pytest.raises(RuntimeError, match="torch"):
            dev_mod.move_to([1, 2, 3], "cpu")

    def test_move_scalar(self) -> None:
        pytest.importorskip("torch")
        from naviertwin.utils.device import move_to

        result = move_to(42, "cpu")
        assert result == 42

    def test_move_module(self) -> None:
        torch = pytest.importorskip("torch")
        import torch.nn as nn

        from naviertwin.utils.device import move_to

        model = nn.Linear(4, 4)
        out = move_to(model, torch.device("cpu"))
        assert isinstance(out, nn.Module)

    def test_memory_no_torch(self, monkeypatch) -> None:
        from naviertwin.utils import device as dev_mod

        real_import = builtins.__import__

        def block(name, *a, **kw):
            if name == "torch":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block)
        info = dev_mod.available_memory_mb()
        assert info["total"] == 0.0
        assert info["cuda"] is False
