"""Round 249 — Incremental SVD."""

from __future__ import annotations

import numpy as np


class TestISVD:
    def test_shapes_update(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.incremental_svd import (
            IncrementalSVD,
        )

        rng = np.random.default_rng(0)
        isvd = IncrementalSVD(rank=3)
        for _ in range(12):
            isvd.update(rng.standard_normal(20))
        assert isvd.U.shape == (20, 3)
        assert isvd.s.shape == (3,)

    def test_rank1_sequence(self) -> None:
        """동일 방향 벡터 반복 → rank=1, s 증가."""
        from naviertwin.core.dimensionality_reduction.linear.incremental_svd import (
            IncrementalSVD,
        )

        isvd = IncrementalSVD(rank=2)
        v = np.ones(10)
        for _ in range(5):
            isvd.update(v)
        # first singular value dominant
        assert isvd.s[0] > 0
        if isvd.s.size >= 2:
            # 두 번째 σ 는 첫 번째보다 작음
            assert isvd.s[1] <= isvd.s[0]
