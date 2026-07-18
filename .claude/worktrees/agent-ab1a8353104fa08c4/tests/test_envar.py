"""Round 344 — EnVar."""

from __future__ import annotations

import numpy as np


class TestEnVar:
    def test_pulls_toward_obs(self) -> None:
        from naviertwin.core.data_assimilation.envar import envar_analysis

        rng = np.random.default_rng(0)
        xb = np.zeros(3)
        ens = rng.standard_normal((20, 3))
        z = np.array([5.0])
        H = np.eye(3)[:1]
        R = 0.01 * np.eye(1)  # tight
        Bs = np.eye(3)
        xa = envar_analysis(xb, ens, z, H, R, Bs, alpha=0.5)
        # x_a[0] should be pulled close to z=5
        assert xa[0] > 1.0

    def test_alpha_extremes(self) -> None:
        """alpha=1.0 → pure static; alpha=0.0 → pure ensemble."""
        from naviertwin.core.data_assimilation.envar import envar_analysis

        rng = np.random.default_rng(0)
        xb = np.zeros(2)
        ens = rng.standard_normal((30, 2))
        z = np.array([1.0])
        H = np.eye(2)[:1]
        R = 0.1 * np.eye(1)
        Bs = np.eye(2)
        x1 = envar_analysis(xb, ens, z, H, R, Bs, alpha=1.0)
        x2 = envar_analysis(xb, ens, z, H, R, Bs, alpha=0.0)
        # different (different B)
        assert not np.allclose(x1, x2)
