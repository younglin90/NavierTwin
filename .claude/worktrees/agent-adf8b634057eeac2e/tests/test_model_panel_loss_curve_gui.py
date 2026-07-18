"""GUI tests for ModelPanel loss curve visibility."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


class DummyOperator:
    """Minimal operator-like object with recorded training losses."""

    train_losses_ = [1.0, 0.5, 0.25]


def test_model_panel_embeds_loss_curve_widget(qtbot) -> None:
    from naviertwin.gui.panels.model_panel import ModelPanel
    from naviertwin.gui.widgets.loss_curve_widget import LossCurveWidget

    panel = ModelPanel()
    qtbot.addWidget(panel)

    assert isinstance(panel._loss_curve, LossCurveWidget)
    assert panel._loss_series == {}


def test_model_panel_updates_loss_curve_from_training_losses(qtbot) -> None:
    from naviertwin.gui.panels.model_panel import ModelPanel

    panel = ModelPanel()
    qtbot.addWidget(panel)

    panel._update_loss_curve("FNO1D", DummyOperator())

    assert panel._loss_series == {"FNO1D": [1.0, 0.5, 0.25]}
    assert panel._loss_curve._series == panel._loss_series
    assert "Loss curve 업데이트" in panel._log_text.toPlainText()
