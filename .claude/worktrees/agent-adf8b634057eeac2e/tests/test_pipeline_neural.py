"""Round 60 — NeuralOperator/PINN/GNN pipelines."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch 필요")


class TestNeuralOperatorPipeline:
    def test_fno1d_end_to_end(self, tmp_path) -> None:
        pytest.importorskip("jinja2")
        from naviertwin.core.digital_twin.pipeline_neural import (
            NeuralOperatorPipeline,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 32, 1)).astype(np.float32)
        Y = np.sin(X).astype(np.float32)

        pipe = NeuralOperatorPipeline(
            kind="fno1d", in_ch=1, out_ch=1,
            modes=4, width=8, n_layers=2, max_epochs=3,
        )
        pipe.fit(X[:15], Y[:15])
        assert len(pipe.state.train_losses) == 3

        metrics = pipe.validate(X[15:], Y[15:])
        assert "rmse" in metrics

        out = pipe.export_report(tmp_path / "nn.html", project="FNO test")
        content = out.read_text(encoding="utf-8")
        assert "FNO1D" in content.upper()

    def test_unet2d(self) -> None:
        from naviertwin.core.digital_twin.pipeline_neural import (
            NeuralOperatorPipeline,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 16, 16, 1)).astype(np.float32)
        Y = X ** 2

        pipe = NeuralOperatorPipeline(
            kind="unet2d", in_ch=1, out_ch=1, base_ch=4, max_epochs=2,
        )
        pipe.fit(X, Y)
        y = pipe.predict(X[:1])
        assert y.shape == (1, 16, 16, 1)

    def test_deeponet(self) -> None:
        from naviertwin.core.digital_twin.pipeline_neural import (
            NeuralOperatorPipeline,
        )

        rng = np.random.default_rng(0)
        m, q = 8, 10
        B = rng.standard_normal((20, m)).astype(np.float32)
        T = rng.standard_normal((q, 1)).astype(np.float32)
        Y = rng.standard_normal((20, q)).astype(np.float32)

        pipe = NeuralOperatorPipeline(
            kind="deeponet", branch_in=m, trunk_in=1,
            hidden=16, latent=8, max_epochs=2,
        )
        pipe.fit(B, Y, trunk_inputs=T)
        y = pipe.predict(B[:3], trunk_inputs=T)
        assert y.shape == (3, q)

    def test_unknown_kind_raises(self) -> None:
        from naviertwin.core.digital_twin.pipeline_neural import (
            NeuralOperatorPipeline,
        )

        pipe = NeuralOperatorPipeline(kind="foobar")
        with pytest.raises(ValueError, match="kind"):
            pipe.fit(np.zeros((1, 4, 1), dtype=np.float32), np.zeros((1, 4, 1), dtype=np.float32))


class TestPINNPipeline:
    def test_pinn_pipeline(self) -> None:
        import torch

        from naviertwin.core.digital_twin.pipeline_neural import PINNPipeline

        def residual(model, x):
            x = x.requires_grad_(True)
            u = model(x)
            du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
            d2u = torch.autograd.grad(du.sum(), x, create_graph=True)[0]
            return d2u + (np.pi ** 2) * torch.sin(np.pi * x)

        pipe = PINNPipeline(in_dim=1, out_dim=1, hidden=32, n_layers=3, max_epochs=300)
        collocation = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(-1, 1)
        bc = {
            "x": np.array([[0.0], [1.0]], dtype=np.float32),
            "u": np.array([[0.0], [0.0]], dtype=np.float32),
        }
        pipe.fit(residual, collocation, bc)
        y = pipe.predict(np.array([[0.5]]))
        assert y.shape == (1, 1)


class TestGNNPipeline:
    def test_gnn_pipeline(self) -> None:
        pytest.importorskip("torch_geometric")
        from naviertwin.core.digital_twin.pipeline_neural import GNNPipeline

        rng = np.random.default_rng(0)
        n_nodes, n_samples = 20, 10
        X = rng.standard_normal((n_samples, n_nodes, 2)).astype(np.float32)
        Y = X ** 2
        edge = np.stack([np.arange(n_nodes), np.roll(np.arange(n_nodes), -1)]).astype(np.int64)

        pipe = GNNPipeline(in_dim=2, out_dim=2, hidden=8, n_layers=2, max_epochs=3)
        pipe.fit(X, Y, edge)
        y = pipe.predict(X[:2])
        assert y.shape == (2, n_nodes, 2)
