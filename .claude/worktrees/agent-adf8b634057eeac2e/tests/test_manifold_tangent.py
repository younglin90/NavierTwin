"""Round 406 — manifold tangent."""

from __future__ import annotations

import numpy as np


class TestManifoldTangent:
    def test_plane_data(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.manifold_tangent import (
            tangent_space,
        )

        rng = np.random.default_rng(0)
        # 2D plane embedded in 5D + small noise
        Z = rng.standard_normal((200, 2))
        E = np.eye(5)[:, :2]  # canonical embedding
        X = Z @ E.T + 0.01 * rng.standard_normal((200, 5))
        T = tangent_space(X, p_idx=0, k=30, dim=2)
        assert T.shape == (5, 2)
        # tangent should align with E (up to rotation): project E onto T
        proj = T.T @ E  # (2, 2)
        # ‖E - T proj‖ small
        rec = T @ proj
        assert np.linalg.norm(E - rec) < 0.3
