"""v3.2.0 ONNX / TorchScript / 보고서 / Undo-Redo / i18n 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch", reason="PyTorch 필요")


class TestONNXExport:
    def test_export_linear_model(self, tmp_path: Path) -> None:
        import torch
        import torch.nn as nn

        from naviertwin.core.export.onnx_export import export_to_onnx

        pytest.importorskip("onnx")

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        sample = torch.randn(1, 4)
        out = export_to_onnx(
            model, sample, tmp_path / "m.onnx",
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_verify_onnx(self, tmp_path: Path) -> None:
        import torch
        import torch.nn as nn

        from naviertwin.core.export.onnx_export import export_to_onnx, verify_onnx

        pytest.importorskip("onnx")

        model = nn.Linear(3, 1)
        sample = torch.randn(1, 3)
        path = export_to_onnx(model, sample, tmp_path / "m.onnx")
        info = verify_onnx(path)
        assert "ir_version" in info
        assert info["graph_inputs"] == ["input"]


class TestTorchScriptExport:
    def test_trace(self, tmp_path: Path) -> None:
        import torch
        import torch.nn as nn

        from naviertwin.core.export.torchscript_export import export_to_torchscript

        model = nn.Linear(3, 1)
        out = export_to_torchscript(
            model, tmp_path / "m.pt",
            sample_input=(torch.randn(1, 3),), mode="trace",
        )
        assert out.exists()

        loaded = torch.jit.load(str(out))
        y = loaded(torch.zeros(1, 3))
        assert y.shape == (1, 1)

    def test_script(self, tmp_path: Path) -> None:
        import torch
        import torch.nn as nn

        from naviertwin.core.export.torchscript_export import export_to_torchscript

        class _M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2.0

        out = export_to_torchscript(_M(), tmp_path / "s.pt", mode="script")
        assert out.exists()


class TestReportGenerator:
    def test_html_generation(self, tmp_path: Path) -> None:
        pytest.importorskip("jinja2")
        from naviertwin.core.report.generator import ReportGenerator

        data = {
            "project": "Test",
            "summary": "POD + Kriging",
            "metrics": {"rmse": 0.012, "r2": 0.98},
            "model_info": {"n_modes": 5, "n_samples": 100},
            "figures": [],
        }
        out = ReportGenerator().render_html(data, tmp_path / "r.html")
        content = out.read_text(encoding="utf-8")
        assert "<html" in content
        assert "POD + Kriging" in content
        assert "0.012" in content


class TestUndoRedo:
    def test_basic(self) -> None:
        from naviertwin.utils.undo_redo import Command, UndoRedoStack

        stack = UndoRedoStack()
        state = {"x": 0}
        cmd = Command(
            do=lambda: state.update(x=state["x"] + 1),
            undo=lambda: state.update(x=state["x"] - 1),
            label="inc",
        )
        stack.execute(cmd)
        assert state["x"] == 1
        stack.undo()
        assert state["x"] == 0
        stack.redo()
        assert state["x"] == 1

    def test_execute_clears_redo(self) -> None:
        from naviertwin.utils.undo_redo import Command, UndoRedoStack

        stack = UndoRedoStack()
        val = [0]
        stack.execute(Command(do=lambda: val.__setitem__(0, 1), undo=lambda: val.__setitem__(0, 0)))
        stack.undo()
        assert stack.can_redo()
        stack.execute(Command(do=lambda: val.__setitem__(0, 2), undo=lambda: val.__setitem__(0, 0)))
        assert not stack.can_redo()

    def test_max_size_cap(self) -> None:
        from naviertwin.utils.undo_redo import Command, UndoRedoStack

        stack = UndoRedoStack(max_size=3)
        for _ in range(5):
            stack.execute(Command(do=lambda: None, undo=lambda: None))
        assert len(stack) == 3


class TestI18n:
    def test_ko_and_en(self) -> None:
        from naviertwin.utils.i18n import Translator

        t = Translator(lang="ko")
        assert t("panel.import") == "불러오기"
        t.set_language("en")
        assert t("panel.import") == "Import"

    def test_unknown_key_returns_fallback(self) -> None:
        from naviertwin.utils.i18n import Translator

        t = Translator(lang="ko")
        assert t("missing.key") == "missing.key"
        assert t("missing.key", default="ABC") == "ABC"

    def test_available_languages_listed(self) -> None:
        from naviertwin.utils.i18n import Translator

        t = Translator(lang="ko")
        langs = t.available_languages()
        assert "ko" in langs and "en" in langs
