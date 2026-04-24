"""Round 41 — POD-Galerkin ROM."""

from __future__ import annotations

import numpy as np
import pytest


class TestPODGalerkin:
    def test_linear_dynamics_recovery(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.pod_galerkin import (
            PODGalerkinROM,
        )

        # x_{k+1} = A x_k + 작은 노이즈
        rng = np.random.default_rng(0)
        A_true = np.diag([0.95, 0.9, 0.8])
        X = np.zeros((3, 100))
        X[:, 0] = np.array([1.0, 1.0, 1.0])
        for k in range(99):
            X[:, k + 1] = A_true @ X[:, k] + 0.001 * rng.standard_normal(3)

        rom = PODGalerkinROM(n_modes=3)
        rom.fit(X)
        assert rom.A_hat_.shape == (3, 3)

        a0 = rom.encode(X[:, 0:1]).ravel()
        traj = rom.rollout(a0, n_steps=20)
        assert traj.shape == (20, 3)
        assert np.all(np.isfinite(traj))

    def test_with_inputs(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.pod_galerkin import (
            PODGalerkinROM,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((5, 40))
        U = rng.standard_normal((2, 40))

        rom = PODGalerkinROM(n_modes=3)
        rom.fit(X, inputs=U)
        assert rom.A_hat_.shape == (3, 3)
        assert rom.B_hat_.shape == (3, 2)

        a0 = rom.encode(X[:, 0:1]).ravel()
        traj = rom.rollout(a0, n_steps=5, inputs=U[:, :5])
        assert traj.shape == (5, 3)

    def test_shape_mismatch(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.pod_galerkin import (
            PODGalerkinROM,
        )

        rom = PODGalerkinROM(n_modes=2)
        with pytest.raises(ValueError):
            rom.fit(np.zeros((5, 20)), inputs=np.zeros((2, 10)))
