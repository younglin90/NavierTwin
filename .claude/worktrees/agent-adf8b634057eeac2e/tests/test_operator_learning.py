"""신경 연산자(FNO/DeepONet/U-Net) 통합 테스트."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch 가 필요합니다")


class TestFNO1D:
    def test_fit_predict_shapes(self) -> None:
        from naviertwin.core.operator_learning.fno.fno import FNO1D

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 32, 1)).astype(np.float32)
        Y = np.sin(X).astype(np.float32)

        op = FNO1D(in_channels=1, out_channels=1, modes=4, width=8, n_layers=2, max_epochs=3)
        op.fit({"inputs": X, "outputs": Y})
        assert op.is_fitted
        assert len(op.train_losses_) == 3

        y_hat = op.predict({"x": X[:5]})
        assert y_hat.shape == (5, 32, 1)

    def test_fit_input_validation(self) -> None:
        from naviertwin.core.operator_learning.fno.fno import FNO1D

        op = FNO1D(in_channels=1, out_channels=1, modes=2, width=4, n_layers=1, max_epochs=1)
        with pytest.raises(ValueError):
            op.fit({"inputs": np.zeros((4, 8)), "outputs": np.zeros((4, 8))})


class TestFNO2D:
    def test_fit_predict_shapes(self) -> None:
        from naviertwin.core.operator_learning.fno.fno import FNO2D

        rng = np.random.default_rng(0)
        X = rng.standard_normal((8, 16, 16, 1)).astype(np.float32)
        Y = X ** 2

        op = FNO2D(
            in_channels=1, out_channels=1, modes1=4, modes2=4,
            width=8, n_layers=2, max_epochs=2,
        )
        op.fit({"inputs": X, "outputs": Y})
        y_hat = op.predict({"x": X[:2]})
        assert y_hat.shape == (2, 16, 16, 1)


class TestDeepONet:
    def test_fit_predict_shapes(self) -> None:
        from naviertwin.core.operator_learning.deeponet.deeponet import DeepONet

        rng = np.random.default_rng(0)
        m, q = 16, 10
        branch_X = rng.standard_normal((40, m)).astype(np.float32)
        trunk_X = rng.standard_normal((q, 2)).astype(np.float32)
        W = rng.standard_normal((m, q)).astype(np.float32)
        Y = np.tanh(branch_X @ W).astype(np.float32)

        op = DeepONet(
            branch_in=m, trunk_in=2, hidden=16, latent=8, max_epochs=3,
        )
        op.fit({"branch_inputs": branch_X, "trunk_inputs": trunk_X, "outputs": Y})
        y_hat = op.predict({"branch_inputs": branch_X[:4], "trunk_inputs": trunk_X})
        assert y_hat.shape == (4, q)

    def test_trunk_cache(self) -> None:
        """predict 시 trunk_inputs 생략하면 학습 좌표 재사용."""
        from naviertwin.core.operator_learning.deeponet.deeponet import DeepONet

        rng = np.random.default_rng(0)
        m, q = 8, 5
        branch_X = rng.standard_normal((20, m)).astype(np.float32)
        trunk_X = rng.standard_normal((q, 1)).astype(np.float32)
        Y = rng.standard_normal((20, q)).astype(np.float32)

        op = DeepONet(branch_in=m, trunk_in=1, hidden=8, latent=4, max_epochs=2)
        op.fit({"branch_inputs": branch_X, "trunk_inputs": trunk_X, "outputs": Y})
        y_hat = op.predict({"branch_inputs": branch_X[:3]})
        assert y_hat.shape == (3, q)


class TestUNet2D:
    def test_fit_predict(self) -> None:
        from naviertwin.core.operator_learning.unet.unet import UNet2D

        rng = np.random.default_rng(0)
        X = rng.standard_normal((6, 16, 16, 1)).astype(np.float32)
        Y = X ** 2

        op = UNet2D(in_channels=1, out_channels=1, base_ch=8, max_epochs=2)
        op.fit({"inputs": X, "outputs": Y})
        y = op.predict({"x": X[:2]})
        assert y.shape == (2, 16, 16, 1)

    def test_unet_rejects_non_divisible_shape(self) -> None:
        from naviertwin.core.operator_learning.unet.unet import UNet2D

        X = np.zeros((2, 15, 15, 1), dtype=np.float32)
        op = UNet2D(in_channels=1, out_channels=1, base_ch=4, max_epochs=1)
        with pytest.raises(ValueError, match="4의 배수"):
            op.fit({"inputs": X, "outputs": X})


class TestNotFittedRaises:
    def test_predict_without_fit(self) -> None:
        from naviertwin.core.operator_learning.fno.fno import FNO1D

        op = FNO1D(in_channels=1, out_channels=1, modes=2, width=4, n_layers=1, max_epochs=1)
        with pytest.raises(RuntimeError, match="fit"):
            op.predict({"x": np.zeros((1, 8, 1))})
