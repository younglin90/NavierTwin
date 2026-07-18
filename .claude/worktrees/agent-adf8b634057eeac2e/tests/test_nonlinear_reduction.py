"""비선형 차원축소(AE/VAE/GNN-AE) 테스트.

PyTorch 필수. PyTorch Geometric 은 optional — 미설치 시 skip.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch 가 필요합니다")


class TestAutoencoder:
    def test_fit_encode_decode_shapes(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.autoencoder import (
            Autoencoder,
        )

        rng = np.random.default_rng(0)
        n_features, n_snap = 40, 30
        X = rng.standard_normal((n_features, n_snap)).astype(np.float64)

        ae = Autoencoder(latent_dim=4, hidden_dims=[32, 8], max_epochs=3)
        ae.fit(X)

        Z = ae.encode(X)
        X_rec = ae.decode(Z)
        assert Z.shape == (n_snap, 4)
        assert X_rec.shape == (n_features, n_snap)
        assert ae.is_fitted
        assert ae.n_components == 4

    def test_reconstruct_smaller_than_input_variance(self) -> None:
        """재구성 오차가 입력 분산 대비 작아야 한다(학습이 진전되었는지)."""
        from naviertwin.core.dimensionality_reduction.nonlinear.autoencoder import (
            Autoencoder,
        )

        rng = np.random.default_rng(1)
        # 저랭크 신호
        U = rng.standard_normal((50, 3))
        V = rng.standard_normal((3, 40))
        X = U @ V

        ae = Autoencoder(latent_dim=3, hidden_dims=[32, 8], max_epochs=30, lr=1e-2)
        ae.fit(X)
        X_rec = ae.reconstruct(X)
        err = float(np.linalg.norm(X - X_rec) / np.linalg.norm(X))
        assert err < 0.5  # 학습이 진행되었다는 정도만 검증


class TestVAE:
    def test_fit_and_sample(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.vae import VAE

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 25))
        vae = VAE(latent_dim=3, hidden_dims=[16, 8], max_epochs=3)
        vae.fit(X)

        mu = vae.encode(X)
        assert mu.shape == (25, 3)
        samples = vae.sample(n_samples=5, seed=1)
        assert samples.shape == (30, 5)


class TestGNNAE:
    def test_gnn_ae_optional(self) -> None:
        pyg = pytest.importorskip("torch_geometric", reason="PyG 필요")
        del pyg
        from naviertwin.core.dimensionality_reduction.nonlinear.gnn_ae import GNNAE

        rng = np.random.default_rng(0)
        n_nodes, n_snap = 40, 20
        X = rng.standard_normal((n_nodes, n_snap))
        points = rng.standard_normal((n_nodes, 3))

        ae = GNNAE(latent_dim=3, hidden_dim=8, k=4, max_epochs=5, points=points)
        ae.fit(X)
        z = ae.encode(X)
        assert z.shape == (n_nodes, 3)
        x_hat = ae.decode(z)
        assert x_hat.shape == (n_nodes, n_snap)


class TestSPODBasic:
    def test_spod_shapes(self) -> None:
        from naviertwin.core.flow_analysis.modal.spod import compute_spod

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 256))
        out = compute_spod(X, dt=0.01, n_fft=64, n_modes=3)
        assert out["frequencies"].shape == (64 // 2 + 1,)
        assert out["eigenvalues"].shape == (64 // 2 + 1, 3)
        assert out["modes"].shape == (50, 64 // 2 + 1, 3)

    def test_spod_raises_on_short_series(self) -> None:
        from naviertwin.core.flow_analysis.modal.spod import compute_spod

        X = np.zeros((10, 8))
        with pytest.raises(ValueError):
            compute_spod(X, dt=0.01, n_fft=64)


class TestTwoPointCorr:
    def test_acf_at_zero_is_one(self) -> None:
        from naviertwin.core.flow_analysis.statistics.two_point_corr import (
            two_point_correlation,
        )

        rng = np.random.default_rng(0)
        u = rng.standard_normal((200, 32))
        R = two_point_correlation(u, axis=1)
        assert R[0] == pytest.approx(1.0, rel=1e-6)

    def test_integral_length_positive(self) -> None:
        from naviertwin.core.flow_analysis.statistics.two_point_corr import (
            integral_length_scale,
            two_point_correlation,
        )

        rng = np.random.default_rng(1)
        # 주기적 신호 → 상관 구조 있음
        x = np.linspace(0, 4 * np.pi, 64)
        u = np.cos(x) + 0.1 * rng.standard_normal((100, 64))
        R = two_point_correlation(u, axis=1)
        L = integral_length_scale(R, dx=(x[1] - x[0]))
        assert L > 0


class TestBoundaryLayer:
    def test_thicknesses_monotonic(self) -> None:
        from naviertwin.core.flow_analysis.boundary_layer.boundary_layer import (
            boundary_layer_thicknesses,
        )

        y = np.linspace(0.0, 0.05, 300)
        U_inf = 10.0
        u = U_inf * (1.0 - np.exp(-y / 0.005))
        out = boundary_layer_thicknesses(y, u, U_inf)
        assert out["delta99"] > 0
        assert out["delta_star"] > out["theta"]
        assert out["H"] >= 1.0


class TestNondim:
    def test_reynolds_and_prandtl(self) -> None:
        from naviertwin.core.flow_analysis.thermofluids.nondim import (
            prandtl,
            reynolds,
        )

        assert float(reynolds(1.0, 10.0, 1.0, 0.01)) == pytest.approx(1000.0)
        assert float(prandtl(1e-3, 1005.0, 0.6)) == pytest.approx(1e-3 * 1005.0 / 0.6)


class TestAugmentation:
    def test_galilean_and_reflect_and_rotation(self) -> None:
        from naviertwin.core.data_augmentation.augmentation import (
            galilean_shift,
            reflect,
            rotate_2d,
        )

        U = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        shifted = galilean_shift(U, np.array([1.0, 0.0, 0.0]))
        assert shifted[0, 0] == pytest.approx(2.0)
        reflected = reflect(U, axis=1)
        assert reflected[0, 1] == pytest.approx(-2.0)
        rotated = rotate_2d(U, angle_rad=np.pi / 2)
        assert rotated[0, 0] == pytest.approx(-2.0, abs=1e-10)
        assert rotated[0, 1] == pytest.approx(1.0, abs=1e-10)
