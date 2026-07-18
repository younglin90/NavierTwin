"""Round 177 — ESN."""

from __future__ import annotations

import numpy as np


class TestESN:
    def test_sine_to_cosine(self) -> None:
        from naviertwin.core.system_id.reservoir import EchoStateNetwork

        t = np.linspace(0, 20, 3000)
        x = np.sin(t)[:, None]
        y = np.cos(t)[:, None]
        esn = EchoStateNetwork(
            n_in=1, n_res=80, n_out=1, spectral_radius=0.9, sparsity=0.1, seed=0,
        ).fit(x, y, ridge=1e-4, discard=100)
        yhat = esn.predict(x)
        corr = float(np.corrcoef(yhat[100:, 0], y[100:, 0])[0, 1])
        assert corr > 0.9

    def test_shapes(self) -> None:
        from naviertwin.core.system_id.reservoir import EchoStateNetwork

        rng = np.random.default_rng(0)
        x = rng.standard_normal((200, 3))
        y = rng.standard_normal((200, 2))
        esn = EchoStateNetwork(n_in=3, n_res=30, n_out=2, seed=0).fit(x, y)
        p = esn.predict(x)
        assert p.shape == (200, 2)
