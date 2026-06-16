"""GUI smoke tests for ModelPanel active-learning candidate selection."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")


class _VarianceSurrogate:
    input_dim = 2
    training_metadata = {
        "explainability": {
            "background": np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=float,
            ),
            "feature_names": ["alpha", "beta"],
            "output_index": 0,
        }
    }

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X.sum(axis=1)

    def predict_with_variance(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        pred = X.sum(axis=1)
        var = (X[:, 0] + 2.0 * X[:, 1]).reshape(-1, 1)
        return pred, var


def test_model_panel_active_learning_controls_render(qtbot) -> None:
    from naviertwin.gui.panels.model_panel import ModelPanel

    panel = ModelPanel()
    qtbot.addWidget(panel)

    assert panel._active_strategy_combo.findText("variance") >= 0
    assert panel._active_strategy_combo.findText("random") >= 0
    assert panel._active_btn.text() == "후보 추천"
    assert panel._active_table.columnCount() == 3
    assert panel._online_update_btn.text() == "온라인 업데이트"
    assert panel._online_table.columnCount() == 3
    assert panel._left_scroll.widgetResizable()
    assert panel._left_scroll.widget() is not None
    assert panel._left_scroll.minimumWidth() == 300
    assert panel._left_scroll.maximumWidth() == 300


def test_model_panel_actions_use_equal_visual_priority(qtbot) -> None:
    from naviertwin.gui.panels.model_panel import ModelPanel

    panel = ModelPanel()
    qtbot.addWidget(panel)

    action_buttons = [
        panel._train_btn,
        panel._active_btn,
        panel._online_update_btn,
        panel._op_train_btn,
        panel._physics_train_btn,
        panel._physics_module_btn,
        panel._advanced_ai_btn,
    ]

    assert all(button.objectName() == "modelActionButton" for button in action_buttons)
    assert panel._train_btn.objectName() != "primaryButton"


def test_model_panel_active_learning_recommends_candidates(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from naviertwin.gui.panels.model_panel import ModelPanel

    panel = ModelPanel()
    qtbot.addWidget(panel)
    panel._surrogate = _VarianceSurrogate()
    panel._active_pool_spin.setValue(10)
    panel._active_k_spin.setValue(2)
    panel._active_low_spin.setValue(-1.0)
    panel._active_high_spin.setValue(1.0)
    emitted: list[dict[str, object]] = []
    captured: dict[str, object] = {}
    panel.active_learning_done.connect(emitted.append)

    def fake_select(model: object, pool: np.ndarray, k: int, strategy: str) -> np.ndarray:
        mean, std = model.predict(pool[:3], return_std=True)
        captured["pool_shape"] = tuple(pool.shape)
        captured["k"] = k
        captured["strategy"] = strategy
        captured["mean_shape"] = tuple(mean.shape)
        captured["std_shape"] = tuple(std.shape)
        return np.array([1, 3], dtype=np.int64)

    monkeypatch.setattr(panel, "_select_next_samples", fake_select)

    panel._run_active_learning()

    assert captured["pool_shape"] == (10, 2)
    assert captured["k"] == 2
    assert captured["strategy"] == "variance"
    assert captured["mean_shape"] == (3,)
    assert captured["std_shape"] == (3,)
    assert panel._active_table.rowCount() == 2
    assert panel._active_table.item(0, 0).text() == "1"
    assert panel._active_table.item(0, 2).text() != "n/a"
    assert emitted and emitted[0]["strategy"] == "variance"
    selected = emitted[0]["selected"]
    assert isinstance(selected, np.ndarray)
    assert selected.shape == (2, 2)
    assert "Active Learning 후보 추천 완료" in panel._log_text.toPlainText()


def test_model_panel_active_learning_requires_surrogate(qtbot) -> None:
    from naviertwin.gui.panels.model_panel import ModelPanel

    panel = ModelPanel()
    qtbot.addWidget(panel)

    panel._run_active_learning()

    assert "학습된 surrogate가 없습니다" in panel._log_text.toPlainText()


def test_model_panel_online_update_uses_recommended_candidate(qtbot) -> None:
    from naviertwin.core.online_learning.online_learning import OnlineKriging
    from naviertwin.gui.panels.model_panel import ModelPanel

    panel = ModelPanel()
    qtbot.addWidget(panel)
    emitted: list[dict[str, object]] = []
    trained: list[tuple[str, object]] = []
    panel.online_learning_done.connect(emitted.append)
    panel.model_trained.connect(lambda name, model: trained.append((name, model)))
    panel._surrogate = _VarianceSurrogate()
    panel._last_active_candidates = np.array([[0.25, 0.75]], dtype=float)
    panel._online_y_spin.setValue(1.0)

    panel._run_online_update()

    assert isinstance(panel._surrogate, OnlineKriging)
    assert panel._online_table.rowCount() == 6
    assert panel._online_table.item(0, 0).text() == "x"
    assert emitted and emitted[0]["buffer_size"] == 4
    assert np.allclose(emitted[0]["x"], np.array([0.25, 0.75]))
    assert trained and trained[0][0] == "online_kriging"
    assert "OnlineKriging 업데이트 완료" in panel._log_text.toPlainText()


def test_model_panel_online_update_requires_background(qtbot) -> None:
    from naviertwin.gui.panels.model_panel import ModelPanel

    class _NoBackgroundSurrogate:
        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.asarray(X, dtype=float).sum(axis=1)

    panel = ModelPanel()
    qtbot.addWidget(panel)
    panel._surrogate = _NoBackgroundSurrogate()

    panel._run_online_update()

    assert "초기화용 background가 없습니다" in panel._log_text.toPlainText()
