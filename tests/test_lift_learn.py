"""Round 405 — Lift & Learn."""

from __future__ import annotations

import numpy as np


class TestLiftLearn:
    def test_lift_shape(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.lift_learn import (
            lift_polynomial,
        )

        X = np.random.default_rng(0).standard_normal((10, 2))
        # X (2 cols) + X² (2) + cross (1) = 5
        XL = lift_polynomial(X)
        assert XL.shape == (10, 5)

    def test_one_dim(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.lift_learn import (
            lift_polynomial,
        )

        X = np.array([[1.0], [2.0], [3.0]])
        XL = lift_polynomial(X)
        # (3, 2): X and X²
        assert XL.shape == (3, 2)
        assert np.allclose(XL[:, 1], [1, 4, 9])
