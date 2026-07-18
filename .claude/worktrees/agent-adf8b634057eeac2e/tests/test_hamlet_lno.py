"""Round 15 — HAMLET + LNO."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch 필요")


class TestHAMLET:
    def test_fit_predict(self) -> None:
        from naviertwin.core.gnn.graph_transformer.hamlet import HAMLET

        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 30, 2)).astype(np.float32)
        Y = np.tanh(X).astype(np.float32)
        pos = rng.standard_normal((30, 3)).astype(np.float32)

        op = HAMLET(
            in_dim=2, out_dim=2, d_model=16, n_heads=2, n_layers=1,
            pos_dim=3, max_epochs=2,
        )
        op.fit({"node_features": X, "outputs": Y, "positions": pos})
        y = op.predict({"x": X[:2]})
        assert y.shape == (2, 30, 2)

    def test_node_count_mismatch(self) -> None:
        from naviertwin.core.gnn.graph_transformer.hamlet import HAMLET

        op = HAMLET(in_dim=1, out_dim=1, d_model=4, n_heads=1, n_layers=1, pos_dim=2, max_epochs=1)
        X = np.zeros((2, 10, 1), dtype=np.float32)
        Y = np.zeros((2, 10, 1), dtype=np.float32)
        pos = np.zeros((5, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="node count"):
            op.fit({"node_features": X, "outputs": Y, "positions": pos})


class TestLNO:
    def test_shapes(self) -> None:
        from naviertwin.core.operator_learning.fno.lno import LNO1D

        rng = np.random.default_rng(0)
        X = rng.standard_normal((6, 32, 1)).astype(np.float32)
        Y = np.cumsum(X, axis=1).astype(np.float32)

        op = LNO1D(
            in_channels=1, out_channels=1, n_poles=4,
            width=8, n_layers=1, max_epochs=2,
        )
        op.fit({"inputs": X, "outputs": Y})
        y = op.predict({"x": X[:2]})
        assert y.shape == (2, 32, 1)
        assert np.all(np.isfinite(y))
