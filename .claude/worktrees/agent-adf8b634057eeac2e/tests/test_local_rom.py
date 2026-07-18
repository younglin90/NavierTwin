"""Round 275 — Local clustered ROM."""

from __future__ import annotations

import numpy as np


class TestLocalROM:
    def test_two_clusters_recover(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.local_rom import LocalROM

        rng = np.random.default_rng(0)
        # cluster 1 around [10,...], cluster 2 around [-10,...]
        X1 = 10.0 * np.ones((6, 20)) + 0.1 * rng.standard_normal((6, 20))
        X2 = -10.0 * np.ones((6, 20)) + 0.1 * rng.standard_normal((6, 20))
        X = np.hstack([X1, X2])
        lr = LocalROM(n_clusters=2, rank=2).fit(X)
        # encode/decode round-trip per cluster
        x = X[:, 0]
        z = lr.encode(x)
        # find which cluster
        d = np.linalg.norm(lr.centers - x[:, None], axis=0)
        j = int(np.argmin(d))
        x_rec = lr.decode(z, cluster=j)
        # reconstruction error small (within cluster basis)
        assert np.linalg.norm(x_rec - x) / np.linalg.norm(x) < 0.1

    def test_shapes(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.local_rom import LocalROM

        rng = np.random.default_rng(2)
        X = rng.standard_normal((10, 50))
        lr = LocalROM(n_clusters=3, rank=4).fit(X)
        assert lr.centers.shape == (10, 3)
        assert len(lr.bases) == 3
        assert lr.bases[0].shape == (10, 4)
