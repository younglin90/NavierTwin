"""Round 211 — HMC."""

from __future__ import annotations

import numpy as np


class TestHMC:
    def test_gaussian(self) -> None:
        from naviertwin.core.uncertainty.hmc import hmc

        def logp(q):
            return -0.5 * float(q @ q)

        def grad(q):
            return -q

        s = hmc(logp, np.zeros(2), n=800, step=0.1, L=20, grad=grad, seed=0)
        m = s.mean(axis=0)
        assert np.all(np.abs(m) < 0.2)
        v = s.var(axis=0)
        assert np.all(np.abs(v - 1.0) < 0.3)

    def test_fd_grad_path(self) -> None:
        from naviertwin.core.uncertainty.hmc import hmc

        def logp(q):
            return -0.5 * float(q @ q)

        s = hmc(logp, np.zeros(1), n=300, step=0.1, L=10, seed=0)
        assert s.shape == (300, 1)
