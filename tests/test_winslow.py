"""Round 302 — Winslow elliptic smoothing."""

from __future__ import annotations

import numpy as np


class TestWinslow:
    def test_perturbation_decays(self) -> None:
        from naviertwin.core.tools.winslow import winslow_smooth

        n = 7
        X = np.tile(np.linspace(0, 1, n), (n, 1))
        Y = np.tile(np.linspace(0, 1, n)[:, None], (1, n))
        X[3, 3] += 0.2
        Y[3, 3] += 0.2
        Xs, Ys = winslow_smooth(X, Y, n_iter=50)
        # interior perturbation decays
        assert abs(Xs[3, 3] - 0.5) < abs(X[3, 3] - 0.5)

    def test_boundary_preserved(self) -> None:
        from naviertwin.core.tools.winslow import winslow_smooth

        n = 5
        X = np.tile(np.linspace(0, 1, n), (n, 1))
        Y = np.tile(np.linspace(0, 1, n)[:, None], (1, n))
        Xs, Ys = winslow_smooth(X, Y, n_iter=20)
        assert np.allclose(Xs[0, :], X[0, :])
        assert np.allclose(Xs[-1, :], X[-1, :])
        assert np.allclose(Ys[:, 0], Y[:, 0])
        assert np.allclose(Ys[:, -1], Y[:, -1])
