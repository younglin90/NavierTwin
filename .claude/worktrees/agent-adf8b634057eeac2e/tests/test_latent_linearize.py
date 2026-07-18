"""Round 402 — latent linearize."""

from __future__ import annotations

import numpy as np


class TestLatentLin:
    def test_recover_known_A(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.latent_linearize import (
            fit_latent_A,
        )

        rng = np.random.default_rng(0)
        A_true = np.array([[0.9, 0.1], [-0.05, 0.95]])
        z = [rng.standard_normal(2)]
        for _ in range(50):
            z.append(A_true @ z[-1])
        Z = np.asarray(z)
        A = fit_latent_A(Z)
        assert np.allclose(A, A_true, atol=1e-8)

    def test_predict(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.latent_linearize import (
            predict_latent,
        )

        A = np.eye(2) * 0.5
        z0 = np.array([1.0, 1.0])
        zs = predict_latent(A, z0, n_steps=3)
        assert zs.shape == (4, 2)
        assert np.allclose(zs[-1], 0.125)
