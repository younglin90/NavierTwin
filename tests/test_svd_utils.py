"""Round 117 — SVD utils."""

from __future__ import annotations

import numpy as np


class TestSVD:
    def test_truncated_reconstruction(self) -> None:
        from naviertwin.core.linalg.svd_utils import (
            low_rank_reconstruct,
            truncated_svd,
        )

        rng = np.random.default_rng(0)
        # rank-3 구조 + 잡음
        L = rng.standard_normal((50, 3))
        R = rng.standard_normal((3, 20))
        A = L @ R
        U, s, Vt = truncated_svd(A, k=3)
        A_hat = low_rank_reconstruct(U, s, Vt)
        assert np.linalg.norm(A - A_hat) / np.linalg.norm(A) < 1e-10

    def test_randomized_matches(self) -> None:
        from naviertwin.core.linalg.svd_utils import (
            low_rank_reconstruct,
            randomized_svd,
            truncated_svd,
        )

        rng = np.random.default_rng(0)
        A = rng.standard_normal((80, 40))
        k = 5
        U1, s1, Vt1 = truncated_svd(A, k)
        U2, s2, Vt2 = randomized_svd(A, k, n_oversamples=20, n_iter=4, seed=0)
        # singular values 근사 일치
        assert np.allclose(np.sort(s1), np.sort(s2), rtol=0.05)
        # reconstruction error 비슷
        e1 = np.linalg.norm(A - low_rank_reconstruct(U1, s1, Vt1))
        e2 = np.linalg.norm(A - low_rank_reconstruct(U2, s2, Vt2))
        assert abs(e1 - e2) / e1 < 0.1

    def test_spectral_and_cond(self) -> None:
        from naviertwin.core.linalg.svd_utils import condition_number, spectral_norm

        A = np.diag([5.0, 2.0, 1.0])
        assert abs(spectral_norm(A) - 5.0) < 1e-12
        assert abs(condition_number(A) - 5.0) < 1e-12
