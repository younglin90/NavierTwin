"""Round 329 — KL expansion."""

from __future__ import annotations

import numpy as np


class TestKL:
    def test_decomp_shapes(self) -> None:
        from naviertwin.core.uncertainty.kl_expansion import kl_decompose

        x = np.linspace(0, 1, 30)
        C = np.exp(-((x[:, None] - x[None, :]) ** 2) / 0.05)
        w, V = kl_decompose(C, n_modes=5)
        assert w.shape == (5,)
        assert V.shape == (30, 5)
        # eigenvalues descending and non-negative
        assert (w >= 0).all()
        assert (np.diff(w) <= 0).all()

    def test_sample_covariance(self) -> None:
        from naviertwin.core.uncertainty.kl_expansion import kl_decompose, kl_sample

        rng = np.random.default_rng(0)
        x = np.linspace(0, 1, 20)
        C = np.exp(-((x[:, None] - x[None, :]) ** 2) / 0.1)
        w, V = kl_decompose(C, n_modes=15)
        samples = np.array([kl_sample(w, V, rng=rng) for _ in range(2000)])
        C_emp = np.cov(samples.T)
        # truncated KL captures most of C
        assert np.linalg.norm(C - C_emp) / np.linalg.norm(C) < 0.3
