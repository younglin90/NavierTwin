"""GUI smoke tests for ExplainabilityPanel."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")


class _DummyDataset:
    n_points = 4
    n_cells = 2


class _DummySurrogate:
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
            "output_index": 1,
        }
    }

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.column_stack([X.sum(axis=1), X[:, 0] - X[:, 1]])


class _NoMetadataModel:
    training_metadata: dict[str, object] = {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float).sum(axis=1)


def test_explainability_panel_runs_kernel_shap(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from naviertwin.core.explainability.shap_explainer import KernelSHAP
    from naviertwin.gui.panels.explainability_panel import ExplainabilityPanel

    panel = ExplainabilityPanel()
    qtbot.addWidget(panel)
    emitted: list[dict[str, object]] = []
    captured: dict[str, object] = {}
    panel.explanation_done.connect(emitted.append)
    panel.set_dataset(_DummyDataset())
    panel.set_model(_DummySurrogate())

    def fake_explain(self: KernelSHAP, X: np.ndarray) -> np.ndarray:
        captured["X_shape"] = tuple(X.shape)
        captured["scalar"] = self.f(np.array([[2.0, 0.5]], dtype=float))
        return np.array([[0.25, -0.5], [0.1, -0.2]], dtype=float)

    monkeypatch.setattr(KernelSHAP, "explain", fake_explain)

    panel._run_shap()

    assert panel._explain_btn.isEnabled()
    assert panel._dataset_label.text() == "4 pts, 2 cells"
    assert captured["X_shape"] == (3, 2)
    assert np.allclose(captured["scalar"], np.array([1.5]))
    assert panel._table.rowCount() == 2
    assert panel._table.item(0, 0).text() == "alpha"
    assert panel._table.item(1, 0).text() == "beta"
    assert panel._table.item(1, 2).text() == "0.35"
    assert emitted and emitted[0]["feature_names"] == ["alpha", "beta"]
    assert "SHAP 완료" in panel._log_text.toPlainText()


def test_explainability_panel_requires_background(qtbot) -> None:
    from naviertwin.gui.panels.explainability_panel import ExplainabilityPanel

    panel = ExplainabilityPanel()
    qtbot.addWidget(panel)
    panel.set_model(_NoMetadataModel())

    assert not panel._explain_btn.isEnabled()
    panel._run_shap()
    assert "background가 없습니다" in panel._log_text.toPlainText()
