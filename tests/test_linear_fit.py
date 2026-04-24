"""Round 112 — linear dynamics fit (DMD-lite)."""

from __future__ import annotations

import numpy as np
import pytest


class TestLinearFit:
    def test_recover_rotation(self) -> None:
        from naviertwin.core.koopman.linear_fit import fit_linear_dynamics

        th = 0.1
        A = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        x = np.array([1.0, 0.0])
        traj = [x]
        for _ in range(200):
            traj.append(A @ traj[-1])
        A_hat = fit_linear_dynamics(np.asarray(traj).T)
        assert np.allclose(A_hat, A, atol=1e-8)

    def test_rollout(self) -> None:
        from naviertwin.core.koopman.linear_fit import rollout_linear

        A = np.eye(2) * 0.5
        x0 = np.array([1.0, 2.0])
        traj = rollout_linear(A, x0, n_steps=3)
        assert traj.shape == (2, 4)
        assert np.allclose(traj[:, -1], 0.5 ** 3 * x0)

    def test_eigenanalysis(self) -> None:
        from naviertwin.core.koopman.linear_fit import eigenanalysis

        A = np.array([[0.9, 0.0], [0.0, 0.8]])
        info = eigenanalysis(A)
        assert info["stable"] is True
        assert set(np.round(info["magnitudes"], 2).tolist()) == {0.8, 0.9}

        A2 = np.array([[1.5, 0.0], [0.0, 0.1]])
        assert eigenanalysis(A2)["stable"] is False

    def test_invalid(self) -> None:
        from naviertwin.core.koopman.linear_fit import fit_linear_dynamics

        with pytest.raises(ValueError):
            fit_linear_dynamics(np.zeros((3, 1)))
