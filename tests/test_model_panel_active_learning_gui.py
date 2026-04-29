"""GUI smoke tests for ModelPanel active-learning candidate selection."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")


class _VarianceSurrogate:
    input_dim = 2

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
