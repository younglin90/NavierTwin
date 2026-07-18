"""Round 26 — BPOD + FastICA."""

from __future__ import annotations

import numpy as np
import pytest


class TestBalancedPOD:
    def test_shapes_and_reconstruction(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.bpod import BalancedPOD

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 15))
        Y = rng.standard_normal((30, 15))
        bpod = BalancedPOD(n_modes=5)
        bpod.fit(X, Y)
        assert bpod.direct_modes_.shape == (30, 5)
        assert bpod.adjoint_modes_.shape == (30, 5)
        c = bpod.project(X[:, 0])
        assert c.shape == (5,)
        x_rec = bpod.reconstruct(c)
        assert x_rec.shape == (30,)

    def test_shape_mismatch(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.bpod import BalancedPOD

        bpod = BalancedPOD(n_modes=2)
        with pytest.raises(ValueError):
            bpod.fit(np.zeros((10, 5)), np.zeros((10, 6)))


class TestFastICA:
    def test_decomposition_shape(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.ica import FastICA

        rng = np.random.default_rng(0)
        t = np.linspace(0, 8, 400)
        s1 = np.sin(2 * t)
        s2 = np.sign(np.sin(3 * t))
        S = np.column_stack([s1, s2])
        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        X = S @ A.T + 0.01 * rng.standard_normal((400, 2))

        ica = FastICA(n_components=2, seed=0)
        S_rec = ica.fit_transform(X)
        assert S_rec.shape == (400, 2)
        assert np.all(np.isfinite(S_rec))
