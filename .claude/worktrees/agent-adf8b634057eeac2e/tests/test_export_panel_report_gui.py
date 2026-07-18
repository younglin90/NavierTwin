"""GUI smoke tests for customer report export from ExportPanel."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("jinja2")


@dataclass
class DummyDataset:
    """Minimal dataset shape needed for report generation."""

    n_points: int = 12
    n_cells: int = 5
    n_time_steps: int = 3
    field_names: list[str] = field(default_factory=lambda: ["U", "p"])
    metadata: dict[str, Any] = field(default_factory=dict)


def _dataset_with_metrics() -> DummyDataset:
    return DummyDataset(
        metadata={
            "project_metadata": {
                "engine": {
                    "surrogate": {
                        "validation_metrics": {
                            "rmse": 0.0123,
                            "r2": 0.9876,
                        }
                    }
                }
            }
        }
    )


def test_export_panel_report_format_visible(qtbot) -> None:
    from naviertwin.gui.panels.export_panel import ExportPanel

    panel = ExportPanel()
    qtbot.addWidget(panel)

    formats = [
        panel._format_combo.itemText(i)
        for i in range(panel._format_combo.count())
    ]
    assert any("보고서" in item for item in formats)
    assert any("ONNX" in item for item in formats)
    assert any("TorchScript" in item for item in formats)
    assert any("FMI" in item or "FMU" in item for item in formats)


def test_export_panel_exports_onnx_model(qtbot, tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    from naviertwin.gui.panels.export_panel import ExportPanel

    panel = ExportPanel()
    qtbot.addWidget(panel)
    emitted: list[str] = []
    calls: list[tuple[object, tuple[int, ...], Path]] = []
    panel.export_done.connect(emitted.append)
    panel.set_model(nn.Linear(3, 2), sample_input=torch.ones(1, 3))
    panel._format_combo.setCurrentIndex(_format_index(panel, "ONNX"))
    out = tmp_path / "model"
    panel._path_edit.setText(str(out))

    def fake_export(model: object, sample_input: object, path: Path) -> Path:
        assert isinstance(sample_input, torch.Tensor)
        calls.append((model, tuple(sample_input.shape), path))
        path.write_bytes(b"onnx")
        return path

    monkeypatch.setattr(panel, "_export_to_onnx", fake_export)

    panel._export()

    expected = tmp_path / "model.onnx"
    assert calls and calls[0][1] == (1, 3)
    assert expected.exists()
    assert emitted == [str(expected)]
    assert "ONNX 모델 저장" in panel._log_text.toPlainText()


def test_export_panel_exports_torchscript_model(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    from naviertwin.gui.panels.export_panel import ExportPanel

    panel = ExportPanel()
    qtbot.addWidget(panel)
    emitted: list[str] = []
    calls: list[tuple[object, tuple[int, ...], Path]] = []
    panel.export_done.connect(emitted.append)
    panel.set_model(nn.Linear(4, 1), sample_input=torch.ones(1, 4))
    panel._format_combo.setCurrentIndex(_format_index(panel, "TorchScript"))
    out = tmp_path / "operator.pt"
    panel._path_edit.setText(str(out))

    def fake_export(model: object, path: Path, sample_input: object) -> Path:
        assert isinstance(sample_input, torch.Tensor)
        calls.append((model, tuple(sample_input.shape), path))
        path.write_bytes(b"torchscript")
        return path

    monkeypatch.setattr(panel, "_export_to_torchscript", fake_export)

    panel._export()

    assert calls and calls[0][1] == (1, 4)
    assert out.exists()
    assert emitted == [str(out)]
    assert "TorchScript 모델 저장" in panel._log_text.toPlainText()


def test_export_panel_exports_fmu_engine(qtbot, tmp_path: Path, monkeypatch) -> None:
    from naviertwin.gui.panels.export_panel import ExportPanel

    panel = ExportPanel()
    qtbot.addWidget(panel)
    emitted: list[str] = []
    calls: list[tuple[object, Path]] = []
    engine = object()
    panel.export_done.connect(emitted.append)
    panel.set_engine(engine)
    panel._format_combo.setCurrentIndex(_format_index(panel, "FMU"))
    out = tmp_path / "digital_twin"
    panel._path_edit.setText(str(out))

    def fake_export(export_engine: object, path: Path) -> Path:
        calls.append((export_engine, path))
        path.write_bytes(b"fmu")
        return path

    monkeypatch.setattr(panel, "_export_to_fmu", fake_export)

    panel._export()

    expected = tmp_path / "digital_twin.fmu"
    assert calls == [(engine, expected)]
    assert expected.exists()
    assert emitted == [str(expected)]
    assert "FMI/FMU 모델 저장" in panel._log_text.toPlainText()


def test_export_panel_model_export_requires_torch_module(qtbot, tmp_path: Path) -> None:
    from naviertwin.gui.panels.export_panel import ExportPanel

    panel = ExportPanel()
    qtbot.addWidget(panel)
    panel._format_combo.setCurrentIndex(_format_index(panel, "ONNX"))
    panel._path_edit.setText(str(tmp_path / "missing.onnx"))

    panel._export()

    assert "내보낼 모델이 없습니다" in panel._log_text.toPlainText()


def test_export_panel_exports_html_report(qtbot, tmp_path: Path) -> None:
    from naviertwin.gui.panels.export_panel import ExportPanel

    panel = ExportPanel()
    qtbot.addWidget(panel)
    emitted: list[str] = []
    panel.export_done.connect(emitted.append)
    panel.set_dataset(_dataset_with_metrics())  # type: ignore[arg-type]
    panel._format_combo.setCurrentIndex(4)
    out = tmp_path / "customer_report.html"
    panel._path_edit.setText(str(out))

    panel._export()

    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "NavierTwin CFD digital-twin report" in html
    assert "0.0123" in html
    assert emitted == [str(out)]


def test_export_panel_report_adds_html_suffix(qtbot, tmp_path: Path) -> None:
    from naviertwin.gui.panels.export_panel import ExportPanel

    panel = ExportPanel()
    qtbot.addWidget(panel)
    panel.set_dataset(DummyDataset())  # type: ignore[arg-type]
    panel._format_combo.setCurrentIndex(4)
    panel._path_edit.setText(str(tmp_path / "handoff"))

    panel._export()

    assert (tmp_path / "handoff.html").exists()


def _format_index(panel: object, needle: str) -> int:
    combo = panel._format_combo
    for idx in range(combo.count()):
        if needle in combo.itemText(idx):
            return idx
    raise AssertionError(f"format not found: {needle}")
