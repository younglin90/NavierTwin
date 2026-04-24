"""Round 209 — GMM."""

from __future__ import annotations

import numpy as np


class TestGMM:
    def test_two_clusters(self) -> None:
        from naviertwin.core.analysis.gmm import GMM

        rng = np.random.default_rng(0)
        c1 = rng.multivariate_normal([0, 0], np.eye(2) * 0.1, size=200)
        c2 = rng.multivariate_normal([5, 5], np.eye(2) * 0.1, size=200)
        X = np.vstack([c1, c2])
        gmm = GMM(k=2, seed=0).fit(X, max_iter=100)
        # 두 센터 복구 (순서 상관 없이)
        m = sorted(gmm.means.tolist(), key=lambda x: x[0])
        assert abs(m[0][0]) < 0.5
        assert abs(m[1][0] - 5.0) < 0.5

    def test_predict(self) -> None:
        from naviertwin.core.analysis.gmm import GMM

        rng = np.random.default_rng(0)
        c1 = rng.multivariate_normal([0, 0], np.eye(2) * 0.05, size=100)
        c2 = rng.multivariate_normal([10, 10], np.eye(2) * 0.05, size=100)
        X = np.vstack([c1, c2])
        gmm = GMM(k=2, seed=0).fit(X)
        labels = gmm.predict(X)
        # 두 클러스터로 분리 (label 값은 임의)
        unique = set(labels.tolist())
        assert len(unique) == 2
