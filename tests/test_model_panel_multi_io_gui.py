"""GUI tests for multi-input/multi-output CFD model training paths."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")


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
        self.cell_data: dict[str, np.ndarray] = {}
        self.n_points = 4
        self.n_cells = 0


class _DummyDataset:
    def __init__(self, offset: float = 0.0) -> None:
        self.mesh = _DummyMesh(offset=offset)
        self.time_steps = [0.0]
        self.field_names = ["p", "U"]
        self.metadata: dict[str, object] = {}
        self.n_points = 4
        self.n_cells = 0
        self.n_time_steps = 1

    def extract_field_snapshots(self, field_name: str) -> np.ndarray:
        raw = self.mesh.point_data[field_name]
        arr = np.asarray(raw, dtype=float)
        if arr.ndim > 1:
            arr = np.linalg.norm(arr, axis=-1)
        return arr.reshape(-1, 1)


def test_model_panel_trains_direct_cfd_surrogate_multi_input_output(
    qtbot,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from naviertwin.gui.panels.model_panel import ModelPanel

    datasets = [_DummyDataset(offset=0.0), _DummyDataset(offset=1.0)]
    params_csv = tmp_path / "params.csv"
    params_csv.write_text(
        "inlet_u,reynolds,ignore\n1.0,100.0,7\n2.0,200.0,8\n",
        encoding="utf-8",
    )

    panel = ModelPanel()
    qtbot.addWidget(panel)
    panel.set_dataset(datasets[0])  # type: ignore[arg-type]
    monkeypatch.setattr(panel, "_load_surrogate_training_datasets", lambda: datasets)

    panel._model_combo.setCurrentText("RBF")
    panel._surrogate_source_combo.setCurrentText("CFD fields (direct)")
    panel._surrogate_params_path = params_csv
    panel._surrogate_params_label.setText(str(params_csv))
    panel._surrogate_param_columns_edit.setText("inlet_u,reynolds")
    panel._surrogate_field_list.item(0).setSelected(True)
    panel._surrogate_field_list.item(1).setSelected(True)
    emitted: list[tuple[str, object]] = []
    panel.model_trained.connect(lambda name, model: emitted.append((name, model)))

    panel._train_model()

    assert emitted and emitted[0][0] == "rbfsurrogate"
    model = emitted[0][1]
    metadata = model.training_metadata
    assert metadata["source"] == "cfd_field_surrogate"
    assert metadata["direct_field_model"] is True
    assert metadata["field_names"] == ["p", "U"]
    assert metadata["parameter_names"] == ["inlet_u", "reynolds"]
    assert metadata["n_params"] == 2
    assert metadata["n_outputs"] == 8
    assert len(metadata["output_fields"]) == 2
    assert model.input_dim == 2
    assert model.output_dim == 8


def test_model_panel_capability_table_documents_multi_io_paths(qtbot) -> None:
    from PySide6.QtWidgets import QGroupBox

    from naviertwin.gui.panels.model_panel import ModelPanel

    panel = ModelPanel()
    qtbot.addWidget(panel)

    table = panel._capability_table
    labels = [table.item(row, 0).text() for row in range(table.rowCount())]

    assert "RBF / Kriging" in labels
    assert "PhysicsNeMo CFD Field" in labels
    assert "FNO/DeepONet/UNet/WNO" in labels
    rbf_row = labels.index("RBF / Kriging")
    assert table.item(rbf_row, 1).text() == "가능"
    assert table.item(rbf_row, 2).text() == "가능"

    group_titles = {group.title() for group in panel.findChildren(QGroupBox)}
    assert "1. 학습 준비" in group_titles
    assert "2A. 모델 학습 - 빠른 Surrogate" in group_titles
    assert "2B. 모델 학습 - Physics AI" in group_titles
    assert "4. 데이터 보강 - Active Learning" in group_titles
    assert "5. 운영 중 업데이트 - Online Update" in group_titles


def test_main_window_connects_direct_field_surrogate_to_twin(qtbot) -> None:
    from naviertwin.core.digital_twin.physics_ai_engine import PhysicsAITwinEngine
    from naviertwin.gui.main_window import MainWindow

    class _DirectSurrogate:
        is_fitted = True
        input_dim = 2
        output_dim = 8
        training_metadata = {
            "source": "cfd_field_surrogate",
            "direct_field_model": True,
            "n_params": 2,
            "n_outputs": 8,
            "parameter_names": ["inlet_u", "reynolds"],
            "output_fields": [
                {"field_name": "p", "display_name": "p", "start": 0, "end": 4},
                {"field_name": "U", "display_name": "U_mag", "start": 4, "end": 8},
            ],
        }

        def predict(self, x: np.ndarray) -> np.ndarray:
            x_arr = np.asarray(x, dtype=float)
            return np.tile(np.arange(8, dtype=float), (x_arr.shape[0], 1))

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    win._on_model_trained("rbfsurrogate", _DirectSurrogate())

    assert isinstance(win._latest_engine, PhysicsAITwinEngine)
    assert win._twin_panel._engine is win._latest_engine
    assert win._twin_panel._n_params_spin.value() == 2
    assert win._twin_panel._param_layout.labelForField(
        win._twin_panel._param_spins[0]
    ).text() == "inlet_u:"
    assert "직접 CFD field surrogate 학습 완료" in win._status_label.text()
