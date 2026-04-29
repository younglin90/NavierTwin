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
    assert panel._assim_method_combo.findText("4D-Var") >= 0
    assert panel._assim_method_combo.findText("Particle Filter") >= 0
    assert panel._assim_method_combo.findText("UKF") >= 0
    assert panel._assim_btn.text() == "동화 quick-check"
    assert panel._design_method_combo.findText("NSGA-II Pareto") >= 0
    assert panel._design_method_combo.findText("SIMP Topology") >= 0
    assert panel._design_btn.text() == "설계 최적화 quick-check"


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


@pytest.mark.parametrize("method", ["4D-Var", "Particle Filter", "UKF"])
def test_twin_panel_runs_assimilation_quick_check(qtbot, method: str) -> None:
    from naviertwin.gui.panels.twin_panel import TwinPanel

    panel = TwinPanel()
    qtbot.addWidget(panel)
    emitted: list[dict[str, object]] = []
    panel.assimilation_done.connect(emitted.append)
    panel._assim_method_combo.setCurrentText(method)
    panel._assim_state_dim_spin.setValue(2)
    panel._assim_steps_spin.setValue(3)
    panel._assim_particles_spin.setValue(80)
    panel._assim_noise_spin.setValue(0.02)

    panel._run_assimilation()

    assert emitted and emitted[0]["method"] == method
    assert emitted[0]["n_state"] == 2
    assert emitted[0]["n_steps"] == 3
    assert float(emitted[0]["error"]) < 1.0
    assert "동화 quick-check 완료" in panel._result_text.toPlainText()
    assert panel._status_label.text() == f"{method} 동화 완료."


def test_twin_panel_runs_nsga2_design_quick_check(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from naviertwin.core.optimization.moo_optimizer import NSGA2
    from naviertwin.gui.panels.twin_panel import TwinPanel

    panel = TwinPanel()
    qtbot.addWidget(panel)
    emitted: list[dict[str, object]] = []
    captured: dict[str, object] = {}
    panel.design_optimization_done.connect(emitted.append)
    panel._design_method_combo.setCurrentText("NSGA-II Pareto")
    panel._design_size_spin.setValue(3)
    panel._design_iter_spin.setValue(2)

    def fake_optimize(self: NSGA2, objective: Callable[[np.ndarray], list[float]]):
        captured["bounds"] = self.bounds.copy()
        captured["n_gen"] = self.n_gen
        captured["objective"] = objective(np.array([0.1, -0.2, 0.3], dtype=float))
        return (
            np.array([[0.1, -0.2, 0.3], [0.2, -0.1, 0.0]], dtype=float),
            np.array([[0.1, 0.5], [0.2, 0.3]], dtype=float),
        )

    monkeypatch.setattr(NSGA2, "optimize", fake_optimize)

    panel._run_design_optimization()

    assert captured["bounds"].shape == (3, 2)
    assert captured["n_gen"] == 2
    assert len(captured["objective"]) == 2
    assert emitted and emitted[0]["pareto_count"] == 2
    assert emitted[0]["n_dims"] == 3
    assert "NSGA-II Pareto" in panel._result_text.toPlainText()


def test_twin_panel_runs_simp_design_quick_check(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from naviertwin.core.optimization import topology_opt
    from naviertwin.gui.panels.twin_panel import TwinPanel

    panel = TwinPanel()
    qtbot.addWidget(panel)
    emitted: list[dict[str, object]] = []
    captured: dict[str, object] = {}
    panel.design_optimization_done.connect(emitted.append)
    panel._design_method_combo.setCurrentText("SIMP Topology")
    panel._design_size_spin.setValue(6)
    panel._design_iter_spin.setValue(4)
    panel._design_volume_spin.setValue(0.4)

    def fake_simp_2d(nx: int, ny: int, vol_frac: float, n_iter: int, **_: object):
        captured["nx"] = nx
        captured["ny"] = ny
        captured["vol_frac"] = vol_frac
        captured["n_iter"] = n_iter
        return np.full((ny, nx), vol_frac, dtype=float)

    monkeypatch.setattr(topology_opt, "simp_2d", fake_simp_2d)

    panel._run_design_optimization()

    assert captured == {"nx": 6, "ny": 3, "vol_frac": 0.4, "n_iter": 4}
    assert emitted and emitted[0]["shape"] == (3, 6)
    assert emitted[0]["volume_fraction"] == pytest.approx(0.4)
    assert "SIMP Topology" in panel._result_text.toPlainText()
