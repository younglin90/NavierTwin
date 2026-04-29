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
    assert not panel._symbolic_btn.isEnabled()
    assert not panel._attention_btn.isEnabled()
    panel._run_shap()
    assert "background가 없습니다" in panel._log_text.toPlainText()


def test_explainability_panel_runs_symbolic_regression(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from naviertwin.core.explainability.symbolic_regression import SymbolicRegressor
    from naviertwin.gui.panels.explainability_panel import ExplainabilityPanel

    panel = ExplainabilityPanel()
    qtbot.addWidget(panel)
    emitted: list[dict[str, object]] = []
    captured: dict[str, object] = {}
    panel.explanation_done.connect(emitted.append)
    panel.set_model(_DummySurrogate())

    def fake_fit(self: SymbolicRegressor, X: np.ndarray, y: np.ndarray) -> None:
        captured["X_shape"] = tuple(X.shape)
        captured["y"] = np.asarray(y, dtype=float)
        self.expression_ = "1.0*x0 + -1.0*x1"
        self.is_fitted = True

    monkeypatch.setattr(SymbolicRegressor, "fit", fake_fit)

    panel._run_symbolic()

    assert panel._symbolic_btn.isEnabled()
    assert captured["X_shape"] == (3, 2)
    assert np.allclose(captured["y"], np.array([0.0, 1.0, -1.0]))
    assert panel._symbolic_text.toPlainText() == "1.0*x0 + -1.0*x1"
    assert emitted and emitted[0]["symbolic_expression"] == "1.0*x0 + -1.0*x1"
    assert "Symbolic 완료" in panel._log_text.toPlainText()


def test_explainability_panel_runs_attention_visualization(qtbot) -> None:
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from naviertwin.gui.panels.explainability_panel import ExplainabilityPanel

    class _AttentionModel:
        def __init__(self) -> None:
            self.attn = nn.MultiheadAttention(
                embed_dim=4, num_heads=2, batch_first=True, dropout=0.0
            )
            self.training_metadata = {
                "attention": {
                    "module_path": "attn",
                    "probe": np.ones((1, 3, 4), dtype=np.float32),
                    "token_names": ["inlet", "wake", "outlet"],
                }
            }

    torch.manual_seed(0)
    panel = ExplainabilityPanel()
    qtbot.addWidget(panel)
    emitted: list[dict[str, object]] = []
    panel.explanation_done.connect(emitted.append)
    panel.set_model(_AttentionModel())

    panel._run_attention()

    assert panel._attention_btn.isEnabled()
    assert panel._attention_source_label.text() == "attn"
    assert panel._attention_matrix_table.rowCount() == 3
    assert panel._attention_matrix_table.columnCount() == 3
    assert panel._attention_matrix_table.horizontalHeaderItem(0).text() == "inlet"
    assert panel._attention_top_table.item(0, 0).text() == "inlet"
    assert emitted
    weights = emitted[0]["attention_weights"]
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (1, 3, 3)
    assert "Attention 완료" in panel._log_text.toPlainText()


def test_explainability_panel_detects_direct_multihead_attention(qtbot) -> None:
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from naviertwin.gui.panels.explainability_panel import ExplainabilityPanel

    torch.manual_seed(0)
    panel = ExplainabilityPanel()
    qtbot.addWidget(panel)
    emitted: list[dict[str, object]] = []
    panel.explanation_done.connect(emitted.append)
    panel.set_model(nn.MultiheadAttention(embed_dim=4, num_heads=2, batch_first=True))
    panel._attention_tokens_spin.setValue(2)

    panel._run_attention()

    assert panel._attention_btn.isEnabled()
    assert panel._attention_source_label.text() == "model"
    assert emitted
    weights = emitted[0]["attention_weights"]
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (1, 2, 2)
