"""Round 11 — Sequential DeepONet + AdaptiveFNO + SpectralRefiner."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch 필요")


class TestSequentialDeepONet:
    def test_basic(self) -> None:
        from naviertwin.core.operator_learning.deeponet.sequential_deeponet import (
            SequentialDeepONet,
        )

        rng = np.random.default_rng(0)
        B = rng.standard_normal((20, 4, 6)).astype(np.float32)
        T = rng.standard_normal((8, 1)).astype(np.float32)
        Y = rng.standard_normal((20, 8)).astype(np.float32)

        op = SequentialDeepONet(
            m=6, history=4, trunk_in=1, hidden=16, latent=8, max_epochs=2,
        )
        op.fit({"branch_inputs": B, "trunk_inputs": T, "outputs": Y})
        y_hat = op.predict({"branch_inputs": B[:3]})
        assert y_hat.shape == (3, 8)

    def test_shape_validation(self) -> None:
        from naviertwin.core.operator_learning.deeponet.sequential_deeponet import (
            SequentialDeepONet,
        )

        op = SequentialDeepONet(m=4, history=3, trunk_in=1, hidden=4, latent=4, max_epochs=1)
        with pytest.raises(ValueError):
            op.fit({
                "branch_inputs": np.zeros((5, 2, 4), dtype=np.float32),  # wrong history
                "trunk_inputs": np.zeros((6, 1), dtype=np.float32),
                "outputs": np.zeros((5, 6), dtype=np.float32),
            })


class TestAdaptiveFNO:
    def test_modes_selected(self) -> None:
        from naviertwin.core.operator_learning.fno.adaptive_fno import AdaptiveFNO1D

        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 32, 1)).astype(np.float32)
        # smooth output (low freq 주도) → 작은 modes 선택
        Y = np.cumsum(X, axis=1)

        op = AdaptiveFNO1D(
            in_channels=1, out_channels=1, width=4,
            energy_threshold=0.9, n_layers=1, max_epochs=2,
        )
        op.fit({"inputs": X, "outputs": Y})
        assert op.modes_selected_ >= 2
        assert op.modes_selected_ <= 17  # rfft bin ≤ N//2 + 1
        y = op.predict({"x": X[:2]})
        assert y.shape == (2, 32, 1)


class TestSpectralRefiner:
    def test_two_stage_training(self) -> None:
        from naviertwin.core.operator_learning.fno.spectral_refiner import (
            SpectralRefiner,
        )

        rng = np.random.default_rng(0)
        Xl = rng.standard_normal((8, 16, 1)).astype(np.float32)
        Yl = np.sin(Xl).astype(np.float32)
        Xh = rng.standard_normal((8, 32, 1)).astype(np.float32)
        Yh = np.sin(Xh).astype(np.float32)

        ref = SpectralRefiner(
            in_channels=1, out_channels=1,
            low_modes=4, high_modes=8, width=8, n_layers=1,
            low_epochs=2, refine_epochs=2,
        )
        ref.fit(Xl, Yl, Xh, Yh)
        assert len(ref.pretrain_losses_) == 2
        assert len(ref.refine_losses_) == 2
        y = ref.predict({"x": Xh[:2]})
        assert y.shape == (2, 32, 1)

    def test_unfitted_raises(self) -> None:
        from naviertwin.core.operator_learning.fno.spectral_refiner import (
            SpectralRefiner,
        )

        ref = SpectralRefiner()
        with pytest.raises(RuntimeError):
            ref.predict({"x": np.zeros((1, 8, 1), dtype=np.float32)})
