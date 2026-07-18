"""Round 595 — WNO1D extended coverage (fit error paths, predict squeeze, pywt block)."""

from __future__ import annotations

import builtins

import numpy as np
import pytest


class TestWNO1DErrorPaths:
    def test_fit_requires_pywt(self, monkeypatch) -> None:
        pytest.importorskip("torch")
        from naviertwin.core.operator_learning.fno.wno import WNO1D

        real_import = builtins.__import__

        def block(name, *a, **kw):
            if name == "pywt":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block)
        op = WNO1D(in_channels=1, out_channels=1, width=4, device="cpu")
        X = np.zeros((4, 16, 1), dtype=np.float32)
        with pytest.raises(RuntimeError, match="pywt"):
            op.fit({"inputs": X, "outputs": X})

    def test_fit_requires_3d_input(self) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("pywt")
        from naviertwin.core.operator_learning.fno.wno import WNO1D

        op = WNO1D(in_channels=1, out_channels=1, width=4, device="cpu")
        X2d = np.zeros((4, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="3D"):
            op.fit({"inputs": X2d, "outputs": X2d})

    def test_predict_2d_squeeze(self) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("pywt")
        from naviertwin.core.operator_learning.fno.wno import WNO1D

        rng = np.random.default_rng(1)
        X = rng.standard_normal((6, 16, 1)).astype(np.float32)
        op = WNO1D(in_channels=1, out_channels=1, width=4,
                   n_layers=1, max_epochs=1, batch_size=4, device="cpu", seed=0)
        op.fit({"inputs": X, "outputs": X})
        # 2D input should be auto-squeezed
        single = X[0]  # (16, 1)
        out = op.predict({"x": single})
        assert out.shape == (16, 1)

    def test_predict_not_fitted(self) -> None:
        pytest.importorskip("torch")
        from naviertwin.core.operator_learning.fno.wno import WNO1D

        op = WNO1D(device="cpu")
        with pytest.raises(RuntimeError):
            op.predict({"x": np.zeros((1, 16, 1))})

    def test_resolve_device_auto(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.operator_learning.fno.wno import WNO1D

        op = WNO1D(device="auto")
        d = op._resolve_device()
        expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert d == expected
