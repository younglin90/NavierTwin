"""Round 274 — Quadratic manifold ROM."""

from __future__ import annotations

import numpy as np


class TestQuadraticManifold:
    def test_shapes(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.quadratic_manifold import (
            QuadraticManifold,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 50))
        qm = QuadraticManifold(rank=4).fit(X)
        Z = qm.encode(X)
        assert Z.shape == (4, 50)
        Xr = qm.decode(Z)
        assert Xr.shape == X.shape

    def test_quadratic_recovery(self) -> None:
        """quadratic data exactly representable → reconstruction near zero."""
        from naviertwin.core.dimensionality_reduction.linear.quadratic_manifold import (
            QuadraticManifold,
        )

        rng = np.random.default_rng(1)
        Phi = np.linalg.qr(rng.standard_normal((20, 2)))[0]
        H = rng.standard_normal((20, 3))  # 2*(2+1)/2 = 3
        m = 30
        Z = rng.standard_normal((2, m))
        Zk = np.array(
            [[Z[0, i]**2, Z[0, i]*Z[1, i], Z[1, i]**2] for i in range(m)]
        ).T
        X = Phi @ Z + H @ Zk
        qm = QuadraticManifold(rank=2).fit(X)
        Xr = qm.decode(qm.encode(X))
        rel_err_qm = np.linalg.norm(X - Xr) / np.linalg.norm(X)
        # linear-only baseline (POD rank 2)
        U, _, _ = np.linalg.svd(X, full_matrices=False)
        Phi_lin = U[:, :2]
        Xr_lin = Phi_lin @ (Phi_lin.T @ X)
        rel_err_lin = np.linalg.norm(X - Xr_lin) / np.linalg.norm(X)
        # quadratic manifold strictly better than linear POD
        assert rel_err_qm < rel_err_lin
