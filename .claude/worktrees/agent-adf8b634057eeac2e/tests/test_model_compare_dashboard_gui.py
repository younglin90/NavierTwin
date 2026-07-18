"""GUI tests for automatic model comparison dashboard updates."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


class DummySurrogate:
    """Minimal trained-model stand-in with validation metrics."""

    training_metadata = {
        "validation_metrics": {
            "rmse": 0.0123,
            "r2": 0.9876,
        }
    }


def test_model_trained_signal_populates_compare_dashboard(qtbot) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    win._on_model_trained("dummy", DummySurrogate())

    assert win._compare_panel is not None
    table = win._compare_panel._table
    assert table.rowCount() == 1
    assert table.item(0, 0).text() == "DummySurrogate"
    assert table.item(0, 1).text() == "0.0123"
    assert table.item(0, 2).text() == "0.9876"


def test_model_panel_preserves_validation_metrics_for_compare(qtbot) -> None:
    from naviertwin.gui.panels.model_panel import ModelPanel

    panel = ModelPanel()
    qtbot.addWidget(panel)
    panel._model_combo.setCurrentText("RBF")
    panel._n_samples_spin.setValue(8)
    panel._n_params_spin.setValue(1)
    panel._n_outputs_spin.setValue(2)

    panel._train_model()

    assert panel._surrogate is not None
    metadata = panel._surrogate.training_metadata
    metrics = metadata["validation_metrics"]
    assert metrics["rmse"] >= 0.0
    assert "r2" in metrics
