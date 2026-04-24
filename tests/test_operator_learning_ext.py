"""v2.0.1 확장 신경 연산자 테스트 (TFNO / PI-DeepONet / MIONet / WNO)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch 가 필요합니다")


class TestTFNO2D:
    def test_fit_predict_param_count_lower_than_dense_fno(self) -> None:
        """TFNO 가 일반 FNO 보다 파라미터가 현저히 적어야 한다."""
        from naviertwin.core.operator_learning.fno.fno import FNO2D
        from naviertwin.core.operator_learning.fno.tfno import TFNO2D

        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 16, 16, 1)).astype(np.float32)
        Y = X ** 2

        tfno = TFNO2D(
            in_channels=1, out_channels=1, modes1=8, modes2=8,
            width=32, rank=4, n_layers=2, max_epochs=2,
        )
        tfno.fit({"inputs": X, "outputs": Y})
        tfno_params = tfno.param_count()

        fno = FNO2D(
            in_channels=1, out_channels=1, modes1=8, modes2=8,
            width=32, n_layers=2, max_epochs=1,
        )
        fno.fit({"inputs": X, "outputs": Y})
        fno_params = sum(p.numel() for p in fno._model.parameters())

        # TFNO 가 덜 파라미터 효율 — 주요 비교 목적
        assert tfno_params < fno_params, (
            f"TFNO params({tfno_params}) should be < FNO({fno_params})"
        )
        y_hat = tfno.predict({"x": X[:2]})
        assert y_hat.shape == (2, 16, 16, 1)


class TestPIDeepONet:
    def test_fit_with_trivial_residual(self) -> None:
        """λ_phys=0 이면 표준 DeepONet 과 동일하게 동작해야 한다."""
        from naviertwin.core.operator_learning.deeponet.pi_deeponet import PIDeepONet

        rng = np.random.default_rng(0)
        m, q = 12, 8
        Bx = rng.standard_normal((30, m)).astype(np.float32)
        Tx = rng.standard_normal((q, 1)).astype(np.float32)
        Y = rng.standard_normal((30, q)).astype(np.float32)

        op = PIDeepONet(
            branch_in=m, trunk_in=1, hidden=16, latent=8,
            max_epochs=2, physics_weight=0.0, residual_fn=None,
        )
        op.fit({"branch_inputs": Bx, "trunk_inputs": Tx, "outputs": Y})
        y_hat = op.predict({"branch_inputs": Bx[:3]})
        assert y_hat.shape == (3, q)
        # phys loss 기록은 존재하되 0
        assert len(op.phys_losses_) == 2
        assert all(v == 0.0 for v in op.phys_losses_)

    def test_residual_fn_invoked(self) -> None:
        """residual_fn 이 있으면 phys_loss > 0 이어야 한다 (nontrivial residual)."""
        import torch

        from naviertwin.core.operator_learning.deeponet.pi_deeponet import PIDeepONet

        rng = np.random.default_rng(1)
        m, q = 8, 6
        Bx = rng.standard_normal((20, m)).astype(np.float32)
        Tx = rng.standard_normal((q, 1)).astype(np.float32)
        Y = rng.standard_normal((20, q)).astype(np.float32)
        phys_coords = rng.standard_normal((5, 1)).astype(np.float32)

        def residual_fn(model_fn: object, coords: torch.Tensor) -> torch.Tensor:
            # 임의의 non-zero 잔차 — branch=0 이면 모델 출력이 bias 근처
            branch_dummy = torch.zeros(1, m, device=coords.device)
            u = model_fn(branch_dummy, coords)
            return u + 1.0  # 상수 오프셋 → 잔차 > 0

        op = PIDeepONet(
            branch_in=m, trunk_in=1, hidden=8, latent=4,
            max_epochs=2, physics_weight=0.1,
            residual_fn=residual_fn, physics_coords=phys_coords,
        )
        op.fit({"branch_inputs": Bx, "trunk_inputs": Tx, "outputs": Y})
        assert op.phys_losses_[-1] > 0


class TestMIONet:
    def test_product_merge(self) -> None:
        from naviertwin.core.operator_learning.deeponet.mionet import MIONet

        rng = np.random.default_rng(0)
        B1 = rng.standard_normal((30, 8)).astype(np.float32)
        B2 = rng.standard_normal((30, 12)).astype(np.float32)
        T = rng.standard_normal((6, 2)).astype(np.float32)
        Y = rng.standard_normal((30, 6)).astype(np.float32)

        op = MIONet(
            branch_in_list=[8, 12], trunk_in=2, hidden=16, latent=8,
            merge="product", max_epochs=2,
        )
        op.fit({"branch_inputs": [B1, B2], "trunk_inputs": T, "outputs": Y})
        y = op.predict({"branch_inputs": [B1[:3], B2[:3]]})
        assert y.shape == (3, 6)

    def test_concat_merge(self) -> None:
        from naviertwin.core.operator_learning.deeponet.mionet import MIONet

        rng = np.random.default_rng(0)
        B1 = rng.standard_normal((20, 4)).astype(np.float32)
        B2 = rng.standard_normal((20, 6)).astype(np.float32)
        T = rng.standard_normal((5, 1)).astype(np.float32)
        Y = rng.standard_normal((20, 5)).astype(np.float32)

        op = MIONet(
            branch_in_list=[4, 6], trunk_in=1, hidden=8, latent=4,
            merge="concat", max_epochs=2,
        )
        op.fit({"branch_inputs": [B1, B2], "trunk_inputs": T, "outputs": Y})
        assert op.is_fitted

    def test_branch_count_validation(self) -> None:
        from naviertwin.core.operator_learning.deeponet.mionet import MIONet

        with pytest.raises(ValueError, match="2 개 이상"):
            MIONet(branch_in_list=[8], trunk_in=1, hidden=4, latent=4, max_epochs=1)


class TestWNO1D:
    def test_fit_predict_with_pywt(self) -> None:
        pytest.importorskip("pywt", reason="PyWavelets 가 필요합니다")
        from naviertwin.core.operator_learning.fno.wno import WNO1D

        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 64, 1)).astype(np.float32)
        Y = X ** 2

        op = WNO1D(
            in_channels=1, out_channels=1, width=8,
            wavelet="db2", level=2, n_layers=2, max_epochs=2,
        )
        op.fit({"inputs": X, "outputs": Y})
        y_hat = op.predict({"x": X[:2]})
        assert y_hat.shape == (2, 64, 1)

    def test_wno_without_pywt_raises(self) -> None:
        from unittest.mock import patch

        from naviertwin.core.operator_learning.fno.wno import WNO1D

        X = np.zeros((2, 32, 1), dtype=np.float32)
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def _blocker(name: str, *args: object, **kwargs: object) -> object:
            if name == "pywt":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        op = WNO1D(in_channels=1, out_channels=1, width=4, level=1, max_epochs=1)
        with patch("builtins.__import__", side_effect=_blocker):
            with pytest.raises(RuntimeError, match="pywt"):
                op.fit({"inputs": X, "outputs": X})
