"""Round 204 — velocity gradient decomposition."""

from __future__ import annotations

import numpy as np
import pytest


class TestVG:
    def test_decompose(self) -> None:
        from naviertwin.core.analysis.velocity_gradient import decompose_J_3x3

        rng = np.random.default_rng(0)
        J = rng.standard_normal((3, 3))
        S, W = decompose_J_3x3(J)
        assert np.allclose(S, S.T)
        assert np.allclose(W, -W.T)
        assert np.allclose(S + W, J)

    def test_invariants_identity(self) -> None:
        from naviertwin.core.analysis.velocity_gradient import invariants_3x3

        J = np.eye(3)
        inv = invariants_3x3(J)
        assert inv["P"] == -3.0
        assert inv["R"] == -1.0

    def test_invalid(self) -> None:
        from naviertwin.core.analysis.velocity_gradient import decompose_J_3x3

        with pytest.raises(ValueError):
            decompose_J_3x3(np.eye(4))

    def test_field_j_2d(self) -> None:
        from naviertwin.core.analysis.velocity_gradient import field_J_2d

        rng = np.random.default_rng(0)
        u = rng.standard_normal((10, 12))
        v = rng.standard_normal((10, 12))
        J = field_J_2d(u, v)
        assert J.shape == (10, 12, 2, 2)
