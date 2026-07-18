"""Round 643 — surrogate gradient + Morris EE + variance decomposition + permutation importance."""

from __future__ import annotations

import numpy as np
import pytest


class TestFDGradient:
    def test_quadratic(self) -> None:
        from naviertwin.core.surrogate.gradient_sensitivity import (
            finite_difference_gradient,
        )

        # f(x) = ‖x‖² → ∇f = 2x
        def f(X):
            return np.sum(X ** 2, axis=1)

        x = np.array([1.0, 2.0, 3.0])
        g = finite_difference_gradient(f, x)
        np.testing.assert_allclose(g, [2.0, 4.0, 6.0], atol=1e-3)

    def test_linear(self) -> None:
        from naviertwin.core.surrogate.gradient_sensitivity import (
            finite_difference_gradient,
        )

        # f(x) = 3x_0 + 2x_1
        def f(X):
            return 3 * X[:, 0] + 2 * X[:, 1]

        g = finite_difference_gradient(f, np.array([1.0, 1.0]))
        np.testing.assert_allclose(g, [3.0, 2.0], atol=1e-5)

    def test_invalid_h(self) -> None:
        from naviertwin.core.surrogate.gradient_sensitivity import (
            finite_difference_gradient,
        )

        with pytest.raises(ValueError, match="h"):
            finite_difference_gradient(lambda x: 0.0, np.array([0.0]), h=0.0)


class TestMorris:
    def test_returns_correct_shape(self) -> None:
        from naviertwin.core.surrogate.gradient_sensitivity import (
            morris_elementary_effects,
        )

        def f(X):
            return X[:, 0] + 2 * X[:, 1] + 0.1 * X[:, 2]

        bounds = np.array([[0.0, 1.0]] * 3)
        mu, sigma = morris_elementary_effects(
            f, bounds, n_trajectories=10, n_levels=4, seed=0,
        )
        assert mu.shape == (3,)
        assert sigma.shape == (3,)
        # x_1이 가장 영향 큼
        assert mu[1] > mu[0] > mu[2]

    def test_invalid_bounds(self) -> None:
        from naviertwin.core.surrogate.gradient_sensitivity import (
            morris_elementary_effects,
        )

        with pytest.raises(ValueError, match="bounds"):
            morris_elementary_effects(
                lambda x: 0.0, bounds=np.array([0.0, 1.0]),
            )

    def test_invalid_n_trajectories(self) -> None:
        from naviertwin.core.surrogate.gradient_sensitivity import (
            morris_elementary_effects,
        )

        with pytest.raises(ValueError, match="n_trajectories"):
            morris_elementary_effects(
                lambda x: 0.0, bounds=np.array([[0.0, 1.0]]),
                n_trajectories=0,
            )

    def test_invalid_n_levels(self) -> None:
        from naviertwin.core.surrogate.gradient_sensitivity import (
            morris_elementary_effects,
        )

        with pytest.raises(ValueError, match="n_levels"):
            morris_elementary_effects(
                lambda x: 0.0, bounds=np.array([[0.0, 1.0]]),
                n_levels=1,
            )


class TestVarianceDecomposition:
    def test_dominant_variable(self) -> None:
        from naviertwin.core.surrogate.gradient_sensitivity import (
            variance_decomposition_1d,
        )

        def f(X):
            # y = 5 x_0 + 0.1 x_1 + 0.1 x_2 → x_0 지배
            return 5 * X[:, 0] + 0.1 * X[:, 1] + 0.1 * X[:, 2]

        bounds = np.array([[0.0, 1.0]] * 3)
        S1 = variance_decomposition_1d(f, bounds, n_samples=2000, seed=0)
        assert S1[0] > S1[1] + 0.5
        assert S1[0] > S1[2] + 0.5

    def test_invalid_bounds(self) -> None:
        from naviertwin.core.surrogate.gradient_sensitivity import (
            variance_decomposition_1d,
        )

        with pytest.raises(ValueError, match="bounds"):
            variance_decomposition_1d(
                lambda x: 0.0, bounds=np.zeros(3),
            )


class TestPermutationImportance:
    def test_relevant_variable(self) -> None:
        from naviertwin.core.surrogate.gradient_sensitivity import (
            permutation_importance,
        )

        def f(X):
            return 10 * X[:, 0] + 0.001 * X[:, 1]  # x_0 매우 중요

        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (200, 2))
        y = f(X)
        importance = permutation_importance(f, X, y, n_repeats=3, seed=0)
        assert importance[0] > importance[1]

    def test_invalid_X_shape(self) -> None:
        from naviertwin.core.surrogate.gradient_sensitivity import (
            permutation_importance,
        )

        with pytest.raises(ValueError, match="shape"):
            permutation_importance(
                lambda x: 0.0, X=np.zeros((10, 3)), y=np.zeros(8),
            )
