"""v4.0.0 Latent Dynamics + Diffusion PDE 테스트."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch 필요")


class TestLatentDynamics:
    def test_shapes_and_rollout(self) -> None:
        from naviertwin.core.time_series.latent_dynamics.latent_dynamics import (
            LatentDynamicsForecaster,
        )

        rng = np.random.default_rng(0)
        seqs = rng.standard_normal((3, 25, 10)).astype(np.float32)
        m = LatentDynamicsForecaster(
            n_features=10, latent=4, hidden=16, field_hidden=16,
            max_epochs=3,
        )
        m.fit({"sequences": seqs, "dt": 0.1})
        assert m.is_fitted
        out = m.predict(seqs[0, 0], n_steps=4)
        assert out.shape == (4, 10)
        assert np.all(np.isfinite(out))

    def test_feature_mismatch_raises(self) -> None:
        from naviertwin.core.time_series.latent_dynamics.latent_dynamics import (
            LatentDynamicsForecaster,
        )

        m = LatentDynamicsForecaster(n_features=5, latent=2, hidden=4, max_epochs=1)
        seqs = np.zeros((2, 10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="n_features"):
            m.fit({"sequences": seqs})


class TestDiffusionPDE:
    def test_sample_shape(self) -> None:
        from naviertwin.core.generative.diffusion_pde.diffusion_pde import (
            DiffusionPDE,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 8)).astype(np.float32)
        m = DiffusionPDE(
            n_features=8, hidden=16, n_steps=10, max_epochs=3,
        )
        m.fit(X)
        samples = m.sample(n_samples=5, seed=1)
        assert samples.shape == (5, 8)
        assert np.all(np.isfinite(samples))

    def test_invalid_n_steps(self) -> None:
        from naviertwin.core.generative.diffusion_pde.diffusion_pde import (
            DiffusionPDE,
        )

        with pytest.raises(ValueError, match="n_steps"):
            DiffusionPDE(n_features=4, n_steps=1)

    def test_feature_mismatch(self) -> None:
        from naviertwin.core.generative.diffusion_pde.diffusion_pde import (
            DiffusionPDE,
        )

        m = DiffusionPDE(n_features=8, n_steps=5, max_epochs=1)
        with pytest.raises(ValueError):
            m.fit(np.zeros((10, 5), dtype=np.float32))

    def test_sample_before_fit_raises(self) -> None:
        from naviertwin.core.generative.diffusion_pde.diffusion_pde import (
            DiffusionPDE,
        )

        m = DiffusionPDE(n_features=4, n_steps=5)
        with pytest.raises(RuntimeError, match="fit"):
            m.sample(n_samples=1)
