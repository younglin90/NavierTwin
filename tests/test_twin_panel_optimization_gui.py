"""GUI smoke tests for TwinPanel optimization controls."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

pytest.importorskip("PySide6")


class _DummySurrogate:
    input_dim = 2


class _DummyEngine:
    reducer_type = "pod"
    surrogate_type = "rbf"
    n_modes = 2
    surrogate = _DummySurrogate()

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def predict(self, params: np.ndarray) -> np.ndarray:
        self.calls.append(np.asarray(params, dtype=float))
        return np.array([params[0, 0] + 2.0 * params[0, 1], 1.0], dtype=float)


def test_twin_panel_optimization_controls_render(qtbot) -> None:
    from naviertwin.gui.panels.twin_panel import TwinPanel

    panel = TwinPanel()
    qtbot.addWidget(panel)

    assert panel._optimizer_combo.findText("SurrogateOptimizer") >= 0
    assert panel._objective_combo.findText("min field mean") >= 0
    assert panel._objective_combo.findText("match target scalar") >= 0
    assert panel._optimize_btn.text() == "최적화 실행"


def test_twin_panel_runs_surrogate_optimizer(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from naviertwin.core.optimization.surrogate_opt import SurrogateOptimizer
    from naviertwin.gui.panels.twin_panel import TwinPanel

    panel = TwinPanel()
    qtbot.addWidget(panel)
    engine = _DummyEngine()
    emitted: list[dict[str, object]] = []
    captured: dict[str, object] = {}

    panel.optimization_done.connect(emitted.append)
    panel.set_engine(engine)
    panel._bound_low_spin.setValue(-1.0)
    panel._bound_high_spin.setValue(2.0)
    panel._n_initial_spin.setValue(3)
    panel._max_iter_spin.setValue(4)

    def fake_minimize(
        self: SurrogateOptimizer,
        objective: Callable[[np.ndarray], float],
    ) -> tuple[np.ndarray, float]:
        captured["bounds"] = self.bounds.copy()
        captured["n_initial"] = self.n_initial
        captured["max_iter"] = self.max_iter
        captured["objective_value"] = objective(np.array([0.2, 0.3]))
        self.y_ = [0.9, 0.2]
        return np.array([0.25, 0.75], dtype=float), 0.2

    monkeypatch.setattr(SurrogateOptimizer, "minimize", fake_minimize)

    panel._run_optimize()

    assert np.allclose(captured["bounds"], np.array([[-1.0, 2.0], [-1.0, 2.0]]))
    assert captured["n_initial"] == 3
    assert captured["max_iter"] == 4
    assert captured["objective_value"] == pytest.approx(0.9)
    assert np.allclose(engine.calls[0], np.array([[0.2, 0.3]]))
    assert emitted and emitted[0]["f_best"] == pytest.approx(0.2)
    assert np.allclose(emitted[0]["x_best"], np.array([0.25, 0.75]))
    assert [spin.value() for spin in panel._param_spins] == [0.25, 0.75]
    assert "최적화 완료" in panel._result_text.toPlainText()
    assert panel._status_label.text() == "최적화 완료."


def test_twin_panel_optimization_requires_engine(qtbot) -> None:
    from naviertwin.gui.panels.twin_panel import TwinPanel

    panel = TwinPanel()
    qtbot.addWidget(panel)

    panel._run_optimize()

    assert "TwinEngine이 없습니다" in panel._result_text.toPlainText()
