"""Round 10 — conditional VAE + IKNO 테스트."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch 필요")


class TestConditionalVAE:
    def test_sample_shape(self) -> None:
        from naviertwin.core.generative.conditional_gen.conditional_gen import (
            ConditionalVAE,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 10)).astype(np.float32)
        C = rng.standard_normal((40, 2)).astype(np.float32)
        cvae = ConditionalVAE(
            n_features=10, cond_dim=2, latent=4, hidden=16, max_epochs=2,
        )
        cvae.fit(X, C)
        s = cvae.sample(C[:4], seed=0)
        assert s.shape == (4, 10)

    def test_mismatched_samples(self) -> None:
        from naviertwin.core.generative.conditional_gen.conditional_gen import (
            ConditionalVAE,
        )

        cvae = ConditionalVAE(n_features=4, cond_dim=2, latent=2, max_epochs=1)
        with pytest.raises(ValueError):
            cvae.fit(np.zeros((5, 4), dtype=np.float32), np.zeros((3, 2), dtype=np.float32))

    def test_sample_before_fit_raises(self) -> None:
        from naviertwin.core.generative.conditional_gen.conditional_gen import (
            ConditionalVAE,
        )

        cvae = ConditionalVAE(n_features=4, cond_dim=2, latent=2)
        with pytest.raises(RuntimeError):
            cvae.sample(np.zeros((1, 2)))


class TestIKNO:
    def test_fit_predict(self) -> None:
        from naviertwin.core.operator_learning.koopman.ikno import IKNO

        rng = np.random.default_rng(0)
        seqs = rng.standard_normal((3, 20, 4)).astype(np.float32)
        m = IKNO(n_features=4, n_blocks=2, hidden=16, max_epochs=3)
        m.fit({"sequences": seqs})
        assert m.is_fitted
        out = m.predict(seqs[0, 0], n_steps=5)
        assert out.shape == (5, 4)
        assert np.all(np.isfinite(out))

    def test_requires_feature_dim_ge_2(self) -> None:
        from naviertwin.core.operator_learning.koopman.ikno import IKNO

        with pytest.raises(ValueError):
            IKNO(n_features=1)
