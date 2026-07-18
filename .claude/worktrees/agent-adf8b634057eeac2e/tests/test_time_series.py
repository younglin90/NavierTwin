"""v2.2.0 시계열/Koopman 모델 테스트."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch 필요")


class TestLSTMForecaster:
    def test_fit_predict_shapes(self) -> None:
        from naviertwin.core.time_series.lstm.lstm import LSTMForecaster

        rng = np.random.default_rng(0)
        seqs = rng.standard_normal((4, 40, 3)).astype(np.float32)
        m = LSTMForecaster(n_features=3, hidden=8, lookback=5, max_epochs=2)
        m.fit({"sequences": seqs})
        assert m.is_fitted
        out = m.predict(seqs[0, :5], n_steps=7)
        assert out.shape == (7, 3)

    def test_requires_enough_timesteps(self) -> None:
        from naviertwin.core.time_series.lstm.lstm import LSTMForecaster

        m = LSTMForecaster(n_features=2, hidden=4, lookback=10, max_epochs=1)
        seqs = np.zeros((2, 5, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="lookback"):
            m.fit({"sequences": seqs})


class TestTransformerForecaster:
    def test_fit_predict(self) -> None:
        from naviertwin.core.time_series.transformer.transformer_ts import (
            TransformerForecaster,
        )

        rng = np.random.default_rng(0)
        seqs = rng.standard_normal((3, 30, 2)).astype(np.float32)
        m = TransformerForecaster(
            n_features=2, d_model=16, n_heads=2, lookback=6, max_epochs=2,
        )
        m.fit({"sequences": seqs})
        out = m.predict(seqs[0, :6], n_steps=4)
        assert out.shape == (4, 2)


class TestNeuralODEForecaster:
    def test_fallback_rk4(self) -> None:
        from naviertwin.core.time_series.neural_ode.neural_ode import (
            NeuralODEForecaster,
        )

        rng = np.random.default_rng(0)
        seqs = rng.standard_normal((3, 20, 2)).astype(np.float32)
        m = NeuralODEForecaster(n_features=2, hidden=8, max_epochs=2)
        m.fit({"sequences": seqs, "dt": 0.1})
        assert m.is_fitted
        out = m.predict(seqs[0, 0], n_steps=3)
        assert out.shape == (3, 2)


class TestKNO:
    def test_learned_koopman_matrix_shape(self) -> None:
        from naviertwin.core.operator_learning.koopman.kno import KNO

        rng = np.random.default_rng(0)
        seqs = rng.standard_normal((4, 25, 3)).astype(np.float32)
        m = KNO(n_features=3, latent=5, hidden=8, max_epochs=2)
        m.fit({"sequences": seqs})
        K = m.koopman_matrix()
        assert K.shape == (5, 5)

    def test_rollout_stable_shape(self) -> None:
        from naviertwin.core.operator_learning.koopman.kno import KNO

        rng = np.random.default_rng(1)
        seqs = rng.standard_normal((2, 15, 2)).astype(np.float32)
        m = KNO(n_features=2, latent=4, hidden=8, max_epochs=2)
        m.fit({"sequences": seqs})
        out = m.predict(seqs[0, 0], n_steps=7)
        assert out.shape == (7, 2)
        assert np.all(np.isfinite(out))
