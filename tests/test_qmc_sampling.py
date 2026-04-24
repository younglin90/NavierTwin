"""Round 38 — QMC samplers 테스트."""

from __future__ import annotations

import numpy as np
import pytest


class TestHalton:
    def test_shape_and_range(self) -> None:
        from naviertwin.core.sampling.qmc import halton

        X = halton(n=64, d=3)
        assert X.shape == (64, 3)
        assert np.all((X >= 0) & (X < 1))

    def test_too_many_dims_raises(self) -> None:
        from naviertwin.core.sampling.qmc import halton

        with pytest.raises(ValueError):
            halton(n=10, d=100)


class TestLatinHypercube:
    def test_shape_and_discrepancy(self) -> None:
        from naviertwin.core.sampling.qmc import latin_hypercube

        L = latin_hypercube(n=50, d=3, seed=0)
        assert L.shape == (50, 3)
        # 각 차원을 10 bin 으로 나누면 n/10 개 bin 채움
        for j in range(3):
            bins = np.floor(L[:, j] * 10).astype(int)
            assert len(set(bins)) >= 9  # 거의 uniform


class TestSobol:
    def test_fallback_to_halton(self) -> None:
        from naviertwin.core.sampling.qmc import sobol

        X = sobol(n=32, d=2, seed=0)
        assert X.shape == (32, 2)
        assert np.all((X >= 0) & (X <= 1))


class TestScaleToBounds:
    def test_scaling(self) -> None:
        from naviertwin.core.sampling.qmc import scale_to_bounds

        U = np.array([[0.0, 0.5], [1.0, 0.25]])
        b = np.array([[-1.0, 1.0], [0.0, 10.0]])
        X = scale_to_bounds(U, b)
        assert X.shape == (2, 2)
        # (0, 0.5) → (-1, 5)
        assert np.allclose(X[0], [-1.0, 5.0])
        # (1, 0.25) → (1, 2.5)
        assert np.allclose(X[1], [1.0, 2.5])
