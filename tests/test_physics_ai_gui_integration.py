"""GUI integration tests for Physics AI model-to-twin workflow."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")


class _DummyPhysicsModel:
    is_fitted = True
    input_dim = 2
    output_dim = 8
    train_losses_ = [1.0, 0.4]
    training_metadata = {
        "source": "cfd_dataset",
        "field_name": "p,U",
        "field_names": ["p", "U"],
        "parameter_names": ["inlet_u", "reynolds"],
        "n_params": 2,
        "n_outputs": 8,
        "output_fields": [
            {
                "field_name": "p",
                "display_name": "p",
                "location": "point",
                "start": 0,
                "end": 4,
                "n_locations": 4,
            },
            {
                "field_name": "U",
                "display_name": "U_mag",
                "location": "point",
                "start": 4,
                "end": 8,
                "n_locations": 4,
            },
        ],
        "validation_metrics": {"rmse": 0.0, "r2": 1.0},
    }

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        base = np.arange(8, dtype=float).reshape(-1, 1)
        return base + x_arr[:, 0].reshape(1, -1)


class _DummyMesh:
    def __init__(self, offset: float = 0.0) -> None:
        self.points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        p = np.array([0.0, 1.0, 2.0, 3.0], dtype=float) + offset
        self.point_data = {
            "p": p,
            "U": np.column_stack([p + 1.0, np.zeros(4), np.zeros(4)]),
        }
        self.cell_data = {}
        self.n_points = 4
        self.n_cells = 0


class _DummyDataset:
    def __init__(self, offset: float = 0.0) -> None:
        self.mesh = _DummyMesh(offset=offset)
        self.time_steps = [0.0]
        self.field_names = ["p", "U"]
        self.metadata = {}
        self.n_points = 4
        self.n_cells = 0
        self.n_time_steps = 1


def test_physics_ai_engine_predicts_directly() -> None:
    from naviertwin.core.digital_twin.physics_ai_engine import PhysicsAITwinEngine

    model = _DummyPhysicsModel()
    engine = PhysicsAITwinEngine.from_fitted_model(
        model,
        model_type="physicsnemo_cfd",
        metadata={"n_params": 2, "n_outputs": 8},
    )

    pred = engine.predict(np.array([[0.5, 100.0]], dtype=float))

    assert engine.reducer_type == "direct_physics_ai"
    assert engine.surrogate_type == "physicsnemo_cfd"
    assert engine.input_dim == 2
    assert pred.shape == (8, 1)
    assert pred[1, 0] == pytest.approx(1.5)


def test_model_panel_exposes_and_emits_physics_ai(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from naviertwin.gui.panels.model_panel import ModelPanel

    panel = ModelPanel()
    qtbot.addWidget(panel)
    panel.set_dataset(_DummyDataset())  # type: ignore[arg-type]
    emitted: list[tuple[str, object]] = []
    panel.model_trained.connect(lambda model_type, model: emitted.append((model_type, model)))
    monkeypatch.setattr(
        panel,
        "_fit_physics_ai_from_dataset",
        lambda **_: _DummyPhysicsModel(),
    )

    assert panel._physics_combo.findText("PhysicsNeMo CFD Field") >= 0
    assert panel._physics_field_list.item(0).text() == "p"
    assert panel._physics_field_list.item(0).isSelected()
    panel._physics_field_list.item(1).setSelected(True)
    assert panel._physics_train_btn.text() == "Physics AI 학습"
    assert panel._physics_module_btn.text() == "PhysicsNeMo Module 저장"

    panel._train_physics_ai()

    assert emitted and emitted[0][0] == "physicsnemo_cfd"
    assert panel._surrogate is emitted[0][1]
    assert "PhysicsNeMo CFD Field" in panel._loss_series
    assert "학습 완료" in panel._log_text.toPlainText()
    metadata = getattr(emitted[0][1], "training_metadata")
    assert metadata["source"] == "cfd_dataset"
    assert metadata["field_name"] == "p,U"
    assert metadata["n_params"] == 2


def test_physicsnemo_cfd_field_model_uses_multiple_inputs_and_outputs() -> None:
    pytest.importorskip("torch")

    from naviertwin.core.physnemo.cfd_field_model import PhysicsNeMoCFDFieldModel

    params = np.array([[1.0, 100.0], [2.0, 200.0]], dtype=float)
    model = PhysicsNeMoCFDFieldModel.from_datasets(
        [_DummyDataset(offset=0.0), _DummyDataset(offset=1.0)],
        field_names=["p", "U"],
        params=params,
        parameter_names=["inlet_u", "reynolds"],
        hidden=8,
        max_epochs=3,
        max_train_points=8,
    )
    pred = model.predict(np.array([[1.5, 150.0]], dtype=float))

    assert model.is_fitted is True
    # v5.0 벡터 성분 보존: p + U_x/U_y/U_z = 채널 4개 × 4점 = 16 (예전엔 |U| 로
    # 뭉개 8이었다).
    assert pred.shape == (16, 1)
    assert model.training_metadata["source"] == "cfd_dataset"
    assert model.training_metadata["field_names"] == ["p", "U"]
    assert model.training_metadata["parameter_names"] == ["inlet_u", "reynolds"]
    assert model.training_metadata["n_locations"] == 4
    assert model.training_metadata["n_params"] == 2
    assert len(model.training_metadata["output_fields"]) == 4  # 채널당 1 spec


def test_main_window_connects_physics_ai_model_to_twin(qtbot) -> None:
    from naviertwin.core.digital_twin.physics_ai_engine import PhysicsAITwinEngine
    from naviertwin.gui.main_window import MainWindow

    model = _DummyPhysicsModel()
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    win._on_model_trained("physicsnemo_cfd", model)

    assert isinstance(win._latest_engine, PhysicsAITwinEngine)
    assert win._twin_panel._engine is win._latest_engine
    assert win._tabs.currentWidget() is win._twin_panel
    assert win._twin_panel._surrogate_combo.currentText() == "physicsnemo_cfd"
    assert win._twin_panel._n_params_spin.value() == 2
    assert win._twin_panel._param_layout.labelForField(
        win._twin_panel._param_spins[0]
    ).text() == "inlet_u:"
    assert "Twin 직접 연결 완료" in win._status_label.text()
