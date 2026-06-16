"""GUI coverage checks for Library capability entries."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def test_reduce_panel_exposes_nonlinear_reducers(qtbot) -> None:
    from naviertwin.gui.panels.reduce_panel import ReducePanel

    panel = ReducePanel()
    qtbot.addWidget(panel)

    methods = [panel._method_combo.itemText(i) for i in range(panel._method_combo.count())]
    assert "Autoencoder" in methods
    assert "VAE" in methods


def test_model_panel_exposes_advanced_ai_quick_checks(qtbot) -> None:
    from naviertwin.gui.panels.model_panel import ModelPanel

    panel = ModelPanel()
    qtbot.addWidget(panel)

    items = [
        panel._advanced_ai_combo.itemText(i)
        for i in range(panel._advanced_ai_combo.count())
    ]
    assert items == ["GNN Surrogate", "LSTM + KNO", "Diffusion PDE"]
    assert panel._advanced_ai_btn.text() == "AI quick-check 실행"


def test_twin_panel_exposes_enkf_uq_and_applied_calculators(qtbot) -> None:
    from naviertwin.gui.panels.twin_panel import TwinPanel

    panel = TwinPanel()
    qtbot.addWidget(panel)

    assert panel._assim_method_combo.findText("EnKF") >= 0
    assert panel._surrogate_combo.findText("physicsnemo_cfd") >= 0
    assert panel._uq_btn.text() == "Monte Carlo UQ 실행"
    assert panel._applied_combo.findText("Fan affinity") >= 0
    assert panel._applied_combo.findText("HVAC duct loss") >= 0
    assert panel._applied_combo.findText("Pump operating point") >= 0


def test_twin_panel_runs_enkf_uq_and_applied_quick_checks(qtbot) -> None:
    from naviertwin.gui.panels.twin_panel import TwinPanel

    panel = TwinPanel()
    qtbot.addWidget(panel)
    assim: list[dict[str, object]] = []
    uq: list[dict[str, object]] = []
    applied: list[dict[str, object]] = []
    panel.assimilation_done.connect(assim.append)
    panel.uq_done.connect(uq.append)
    panel.applied_done.connect(applied.append)

    panel._assim_method_combo.setCurrentText("EnKF")
    panel._assim_state_dim_spin.setValue(2)
    panel._assim_steps_spin.setValue(2)
    panel._assim_particles_spin.setValue(40)
    panel._run_assimilation()

    panel._uq_samples_spin.setValue(32)
    panel._run_monte_carlo_uq()

    panel._applied_combo.setCurrentText("Fan affinity")
    panel._run_applied_calculator()

    assert assim and assim[0]["method"] == "EnKF"
    assert uq and uq[0]["method"] == "Monte Carlo UQ"
    assert applied and applied[0]["calculator"] == "Fan affinity"
