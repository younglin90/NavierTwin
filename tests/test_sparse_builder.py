"""Round 118 — 희소 행렬 빌더."""

from __future__ import annotations

import numpy as np
import pytest


class TestSparse:
    def test_laplacian_1d(self) -> None:
        pytest.importorskip("scipy")
        from naviertwin.core.linalg.sparse_builder import laplacian_1d

        L = laplacian_1d(5, h=1.0).toarray()
        assert L.shape == (5, 5)
        assert L[0, 0] == -2.0 and L[0, 1] == 1.0
        assert L[-1, -1] == -2.0

    def test_periodic(self) -> None:
        pytest.importorskip("scipy")
        from naviertwin.core.linalg.sparse_builder import laplacian_1d

        L = laplacian_1d(4, h=1.0, boundary="periodic").toarray()
        assert L[0, -1] == 1.0
        assert L[-1, 0] == 1.0

    def test_laplacian_2d_shape(self) -> None:
        pytest.importorskip("scipy")
        from naviertwin.core.linalg.sparse_builder import laplacian_2d

        L = laplacian_2d(4, 5)
        assert L.shape == (20, 20)

    def test_coo_to_csr(self) -> None:
        pytest.importorskip("scipy")
        from naviertwin.core.linalg.sparse_builder import coo_to_csr

        M = coo_to_csr(
            np.array([0, 1, 2]), np.array([2, 0, 1]),
            np.array([1.0, 2.0, 3.0]), (3, 3),
        ).toarray()
        assert M[0, 2] == 1.0
        assert M[1, 0] == 2.0
        assert M[2, 1] == 3.0
