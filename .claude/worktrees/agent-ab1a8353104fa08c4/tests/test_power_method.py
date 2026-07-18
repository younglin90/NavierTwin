"""Round 189 — power iteration."""

from __future__ import annotations

import numpy as np


class TestPower:
    def test_dominant(self) -> None:
        from naviertwin.core.linalg.power_method import power_iteration

        A = np.diag([10.0, 3.0, 1.0])
        lam, v = power_iteration(A, n_iter=300)
        assert abs(lam - 10.0) < 1e-8
        # eigenvector ≈ e_0
        assert abs(abs(v[0]) - 1.0) < 1e-6

    def test_inverse(self) -> None:
        from naviertwin.core.linalg.power_method import inverse_power

        A = np.diag([10.0, 3.0, 1.0])
        # shift near 3 → returns eigenvalue closest to 3
        lam, _ = inverse_power(A, shift=3.1, n_iter=100)
        assert abs(lam - 3.0) < 1e-6

    def test_rayleigh(self) -> None:
        from naviertwin.core.linalg.power_method import rayleigh_quotient

        A = np.array([[2.0, 0.0], [0.0, 3.0]])
        assert abs(rayleigh_quotient(A, np.array([1, 0])) - 2.0) < 1e-12
        assert abs(rayleigh_quotient(A, np.array([0, 1])) - 3.0) < 1e-12

    def test_random(self) -> None:
        from naviertwin.core.linalg.power_method import power_iteration

        rng = np.random.default_rng(0)
        M = rng.standard_normal((10, 10))
        A = M @ M.T  # SPD
        lam, _ = power_iteration(A, n_iter=500)
        evs = np.linalg.eigvalsh(A)
        assert abs(lam - evs[-1]) < 1e-6
