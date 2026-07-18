"""Round 278 — Randomized DMD."""

from __future__ import annotations

import numpy as np


class TestRDMD:
    def test_shapes(self) -> None:
        from naviertwin.core.system_id.randomized_dmd import randomized_dmd

        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 25))
        evals, modes = randomized_dmd(X, rank=4)
        assert evals.shape == (4,)
        assert modes.shape == (40, 4)

    def test_oscillation_eigenvalue_unit_circle(self) -> None:
        """Pure oscillation x_t = R(θ) x_{t-1} → |λ| ≈ 1."""
        from naviertwin.core.system_id.randomized_dmd import randomized_dmd

        theta = 0.3
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        x = np.array([1.0, 0.0])
        snaps = [x]
        for _ in range(30):
            snaps.append(R @ snaps[-1])
        X = np.column_stack(snaps)
        evals, _ = randomized_dmd(X, rank=2, oversamp=2)
        mags = np.abs(evals)
        assert np.allclose(np.sort(mags), [1.0, 1.0], atol=0.05)
