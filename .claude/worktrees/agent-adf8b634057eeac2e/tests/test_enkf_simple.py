"""Round 155 — Simple EnKF."""

from __future__ import annotations

import numpy as np


class TestEnKF:
    def test_posterior_mean_moves_toward_obs(self) -> None:
        from naviertwin.core.data_assimilation.enkf_simple import EnKFSimple

        rng = np.random.default_rng(0)
        N = 200
        ens = rng.multivariate_normal([0.0, 0.0], np.eye(2), size=N)
        kf = EnKFSimple(H=np.eye(2), R=np.eye(2) * 0.1)
        z = np.array([3.0, -2.0])
        ens_post = kf.update(ens, z, rng=rng)
        m_prior = ens.mean(axis=0)
        m_post = ens_post.mean(axis=0)
        assert np.linalg.norm(m_post - z) < np.linalg.norm(m_prior - z)

    def test_partial_obs(self) -> None:
        from naviertwin.core.data_assimilation.enkf_simple import EnKFSimple

        rng = np.random.default_rng(0)
        N = 200
        ens = rng.multivariate_normal([0.0, 0.0, 0.0], np.eye(3), size=N)
        # 첫 두 차원만 관측
        H = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        kf = EnKFSimple(H=H, R=np.eye(2) * 0.05)
        z = np.array([2.0, 2.0])
        ens_post = kf.update(ens, z, rng=rng)
        m = ens_post.mean(axis=0)
        # 관측 차원만 이동
        assert abs(m[0] - 2.0) < 0.5
        assert abs(m[1] - 2.0) < 0.5
