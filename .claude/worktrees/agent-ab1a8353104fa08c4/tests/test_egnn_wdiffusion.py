"""Round 14 — EGNN equivariance + WaveletDiffusionNO."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch 필요")


class TestEGNN:
    def test_translation_equivariance(self) -> None:
        """좌표 평행이동에 대해 h 는 불변, x 는 동일하게 이동되어야."""
        from naviertwin.core.equivariant.physics_embedded.physics_embedded_gnn import (
            EGNN,
        )

        rng = np.random.default_rng(0)
        x = rng.standard_normal((8, 3)).astype(np.float32)
        h = rng.standard_normal((8, 4)).astype(np.float32)
        model = EGNN(feat_dim=4, n_layers=2, hidden=16, coord_update_scale=0.01)

        x1, h1 = model.forward(x, h)
        shift = np.array([5.0, -3.0, 2.0], dtype=np.float32)
        x2, h2 = model.forward(x + shift, h)

        # h 는 동일해야 (불변)
        assert np.allclose(h1, h2, atol=1e-4)
        # x2 ≈ x1 + shift (동등)
        assert np.allclose(x2 - shift, x1, atol=1e-4)

    def test_rotation_equivariance(self) -> None:
        """2D 회전에 대해서도 좌표는 회전, h 는 불변."""
        from naviertwin.core.equivariant.physics_embedded.physics_embedded_gnn import (
            EGNN,
        )

        rng = np.random.default_rng(0)
        # 2D 평면에 3D 좌표 (z=0)
        x2d = rng.standard_normal((6, 2)).astype(np.float32)
        x = np.column_stack([x2d, np.zeros(6, dtype=np.float32)])
        h = rng.standard_normal((6, 3)).astype(np.float32)
        model = EGNN(feat_dim=3, n_layers=2, hidden=8, coord_update_scale=0.01)

        x1, h1 = model.forward(x, h)

        theta = np.pi / 5
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ], dtype=np.float32)
        x_rot = x @ R.T
        x2, h2 = model.forward(x_rot, h)

        # h 불변
        assert np.allclose(h1, h2, atol=1e-4)
        # x2 ≈ x1 @ R^T
        assert np.allclose(x2, x1 @ R.T, atol=1e-3)


class TestWaveletDiffusion:
    def test_sample_round_trip(self) -> None:
        pytest.importorskip("pywt")
        from naviertwin.core.generative.wavelet_diffusion.wavelet_diffusion_no import (
            WaveletDiffusionNO,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 32)).astype(np.float32)
        m = WaveletDiffusionNO(
            n_features=32, wavelet="db2", level=1,
            n_steps=6, max_epochs=2, hidden=16,
        )
        m.fit(X)
        s = m.sample(3, seed=0)
        assert s.shape == (3, 32)
        assert np.all(np.isfinite(s))

    def test_unfitted_raises(self) -> None:
        from naviertwin.core.generative.wavelet_diffusion.wavelet_diffusion_no import (
            WaveletDiffusionNO,
        )

        m = WaveletDiffusionNO(n_features=16, n_steps=4)
        with pytest.raises(RuntimeError):
            m.sample(1)
