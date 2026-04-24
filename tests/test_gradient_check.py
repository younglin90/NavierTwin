"""Round 72 — gradient check / Jacobian."""

from __future__ import annotations

import numpy as np
import pytest


class TestFDGradient:
    def test_polynomial(self) -> None:
        from naviertwin.utils.gradient_check import finite_difference_gradient

        def f(x: np.ndarray) -> float:
            return float(x[0] ** 2 + 2 * x[1] + 3 * x[2] ** 3)

        g = finite_difference_gradient(f, np.array([1.0, 1.0, 1.0]))
        assert np.allclose(g, [2.0, 2.0, 9.0], atol=1e-5)

    def test_sin_cos(self) -> None:
        from naviertwin.utils.gradient_check import finite_difference_gradient

        def f(x: np.ndarray) -> float:
            return float(np.sin(x[0]) + np.cos(x[1]))

        g = finite_difference_gradient(f, np.array([0.5, 0.5]))
        assert abs(g[0] - np.cos(0.5)) < 1e-5
        assert abs(g[1] + np.sin(0.5)) < 1e-5


class TestJacobian:
    def test_linear_jacobian(self) -> None:
        from naviertwin.utils.gradient_check import finite_difference_jacobian

        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        def f(x: np.ndarray) -> np.ndarray:
            return A @ x

        J = finite_difference_jacobian(f, np.array([1.0, 1.0]))
        assert J.shape == (3, 2)
        assert np.allclose(J, A, atol=1e-5)


class TestGradientCheck:
    def test_correct_analytic(self) -> None:
        from naviertwin.utils.gradient_check import gradient_check

        def f(x: np.ndarray) -> float:
            return float(np.sum(x ** 2))

        def g(x: np.ndarray) -> np.ndarray:
            return 2 * x

        res = gradient_check(f, g, np.array([1.0, 2.0, 3.0]))
        assert res["rel_error"] < 1e-4
        assert res["ok"] == 1.0

    def test_buggy_analytic(self) -> None:
        from naviertwin.utils.gradient_check import gradient_check

        def f(x: np.ndarray) -> float:
            return float(np.sum(x ** 2))

        def bad_g(x: np.ndarray) -> np.ndarray:
            return x  # wrong: should be 2x

        res = gradient_check(f, bad_g, np.array([1.0, 2.0, 3.0]))
        assert res["rel_error"] > 0.01
        assert res["ok"] == 0.0


class TestTorchAutograd:
    def test_gradient_matches_fd(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.utils.gradient_check import (
            finite_difference_gradient,
            torch_autograd_gradient,
        )

        def f_torch(x: torch.Tensor) -> torch.Tensor:
            return (x ** 2).sum()

        def f_np(x: np.ndarray) -> float:
            return float(np.sum(x ** 2))

        x0 = np.array([1.0, -2.0, 3.0])
        g_torch = torch_autograd_gradient(f_torch, x0)
        g_fd = finite_difference_gradient(f_np, x0)
        assert np.allclose(g_torch, g_fd, atol=1e-4)
