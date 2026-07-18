"""Round 32 — UNet3D + CNNAE."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch 필요")


class TestUNet3D:
    def test_shapes(self) -> None:
        from naviertwin.core.operator_learning.unet.unet3d import UNet3D

        rng = np.random.default_rng(0)
        X = rng.standard_normal((2, 8, 8, 8, 1)).astype(np.float32)
        Y = X ** 2
        net = UNet3D(in_channels=1, out_channels=1, base_ch=4, max_epochs=1)
        net.fit({"inputs": X, "outputs": Y})
        y = net.predict({"x": X[:1]})
        assert y.shape == (1, 8, 8, 8, 1)

    def test_non_div4_raises(self) -> None:
        from naviertwin.core.operator_learning.unet.unet3d import UNet3D

        net = UNet3D(base_ch=4, max_epochs=1)
        X = np.zeros((1, 7, 8, 8, 1), dtype=np.float32)
        with pytest.raises(ValueError, match="4의 배수"):
            net.fit({"inputs": X, "outputs": X})


class TestCNNAE:
    def test_encode_decode(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.cnn_ae import CNNAE

        rng = np.random.default_rng(0)
        X = rng.standard_normal((6, 16, 16, 1)).astype(np.float32)
        ae = CNNAE(H=16, W=16, channels=1, latent_dim=4, base_ch=8, max_epochs=2)
        ae.fit(X)
        z = ae.encode(X)
        assert z.shape == (6, 4)
        x_rec = ae.decode(z)
        assert x_rec.shape == (6, 16, 16, 1)

    def test_non_div4_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.cnn_ae import CNNAE

        with pytest.raises(ValueError):
            CNNAE(H=14, W=16)
