"""Round 12 — L-DeepONet + PI-Latent-NO."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch 필요")


class TestLDeepONet:
    def test_shapes(self) -> None:
        from naviertwin.core.operator_learning.latent_operator.l_deeponet import (
            LDeepONet,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 30)).astype(np.float32)
        Y = np.tanh(X).astype(np.float32)
        op = LDeepONet(n_features=30, latent=4, hidden=16, max_epochs=3)
        op.fit({"inputs": X, "outputs": Y})
        y_hat = op.predict({"x": X[:2]})
        assert y_hat.shape == (2, 30)

    def test_feature_mismatch(self) -> None:
        from naviertwin.core.operator_learning.latent_operator.l_deeponet import (
            LDeepONet,
        )

        op = LDeepONet(n_features=10, latent=2, hidden=8, max_epochs=1)
        with pytest.raises(ValueError):
            op.fit({
                "inputs": np.zeros((3, 5), dtype=np.float32),
                "outputs": np.zeros((3, 5), dtype=np.float32),
            })


class TestPILatentNO:
    def test_residual_recorded(self) -> None:
        import torch

        from naviertwin.core.operator_learning.latent_operator.pi_latent_no import (
            PILatentNO,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((15, 20)).astype(np.float32)
        Y = np.tanh(X).astype(np.float32)

        def residual(z_in: torch.Tensor, z_out: torch.Tensor, dec) -> torch.Tensor:
            return z_out - z_in + 1.0  # 상수 offset → 잔차 > 0

        op = PILatentNO(
            n_features=20, latent=3, hidden=16, max_epochs=3,
            residual_fn=residual, physics_weight=0.1,
        )
        op.fit({"inputs": X, "outputs": Y})
        assert len(op.data_losses_) == 3
        assert len(op.phys_losses_) == 3
        assert op.phys_losses_[-1] > 0

    def test_no_residual_zero(self) -> None:
        from naviertwin.core.operator_learning.latent_operator.pi_latent_no import (
            PILatentNO,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 8)).astype(np.float32)
        Y = X.copy()
        op = PILatentNO(
            n_features=8, latent=2, hidden=8, max_epochs=2,
            residual_fn=None, physics_weight=1.0,
        )
        op.fit({"inputs": X, "outputs": Y})
        assert all(v == 0.0 for v in op.phys_losses_)
