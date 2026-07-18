"""Round 585 — WNO1D init + lightweight build coverage (was 28%)."""

from __future__ import annotations

import numpy as np
import pytest


class TestWNO:
    def test_init_attributes(self) -> None:
        from naviertwin.core.operator_learning.fno.wno import WNO1D

        op = WNO1D(in_channels=2, out_channels=3, width=16,
                     wavelet="db2", level=3, n_layers=4, max_epochs=5,
                     batch_size=4, lr=5e-4, device="cpu", seed=42)
        assert op.in_channels == 2
        assert op.out_channels == 3
        assert op.width == 16
        assert op.wavelet == "db2"
        assert op.level == 3
        assert op.n_layers == 4
        assert op.max_epochs == 5
        assert op.batch_size == 4
        assert op.lr == 5e-4
        assert op.seed == 42
        assert op.train_losses_ == []

    def test_resolve_device_cpu(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.operator_learning.fno.wno import WNO1D

        op = WNO1D(device="cpu")
        d = op._resolve_device()
        assert d == torch.device("cpu")

    def test_fit_smoke_or_skip(self) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("pywt")
        from naviertwin.core.operator_learning.fno.wno import WNO1D

        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 32, 1)).astype(np.float32)
        Y = X ** 2
        op = WNO1D(in_channels=1, out_channels=1, width=4,
                     n_layers=1, max_epochs=1, batch_size=2, device="cpu", seed=0)
        op.fit({"inputs": X, "outputs": Y})
        out = op.predict({"x": X[:2]})
        assert out.shape == (2, 32, 1)
