"""Round 56 — pyMOR POD/DEIM/Gram-Schmidt 래퍼."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pymor", reason="pymor 필요")


class TestPymorPOD:
    def test_basis_shapes(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.pymor_pod import pymor_pod

        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 15))
        modes, svals = pymor_pod(X, modes=5)
        assert modes.shape[0] == 40
        assert modes.shape[1] <= 5
        assert svals.size == modes.shape[1]
        # 특이값 내림차순
        assert all(svals[i] >= svals[i + 1] for i in range(len(svals) - 1))

    def test_orthonormal_basis(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.pymor_pod import pymor_pod

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 10))
        modes, _ = pymor_pod(X, modes=4)
        gram = modes.T @ modes
        assert np.allclose(gram, np.eye(modes.shape[1]), atol=1e-8)


class TestPymorDEIM:
    def test_interpolation_indices(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.pymor_pod import pymor_deim

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 20))
        res = pymor_deim(X, modes=5)
        assert "interpolation_indices" in res
        assert "collateral_basis" in res
        k = res["collateral_basis"].shape[1]
        assert len(res["interpolation_indices"]) == k
        # 인덱스 범위 체크
        assert np.all(res["interpolation_indices"] >= 0)
        assert np.all(res["interpolation_indices"] < 50)


class TestGramSchmidt:
    def test_orthogonal(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.pymor_pod import (
            pymor_gram_schmidt,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((25, 8))
        Q = pymor_gram_schmidt(X)
        gram = Q.T @ Q
        assert np.allclose(gram, np.eye(Q.shape[1]), atol=1e-8)
