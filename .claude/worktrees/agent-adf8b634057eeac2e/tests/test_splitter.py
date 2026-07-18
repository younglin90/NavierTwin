"""Round 89 — dataset splitter + k-fold."""

from __future__ import annotations

import numpy as np
import pytest


class TestSplit:
    def test_sizes(self) -> None:
        from naviertwin.core.preprocessing.splitter import split_indices

        tr, va, te = split_indices(100, val=0.2, test=0.1, seed=0)
        assert len(tr) == 70
        assert len(va) == 20
        assert len(te) == 10
        # 전부 유니크
        assert len(set(tr) | set(va) | set(te)) == 100

    def test_reproducible(self) -> None:
        from naviertwin.core.preprocessing.splitter import split_indices

        a = split_indices(50, val=0.3, seed=42)
        b = split_indices(50, val=0.3, seed=42)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)

    def test_invalid(self) -> None:
        from naviertwin.core.preprocessing.splitter import split_indices

        with pytest.raises(ValueError):
            split_indices(10, val=0.7, test=0.5)

    def test_split_snapshots(self) -> None:
        from naviertwin.core.preprocessing.splitter import split_snapshots

        X = np.arange(80).reshape(4, 20).astype(float)
        out = split_snapshots(X, val=0.25, test=0.1, seed=0)
        assert out["train"].shape[1] + out["val"].shape[1] + out["test"].shape[1] == 20

    def test_kfold(self) -> None:
        from naviertwin.core.preprocessing.splitter import k_fold_indices

        folds = k_fold_indices(30, k=5, seed=0)
        assert len(folds) == 5
        for tr, va in folds:
            assert len(set(tr) & set(va)) == 0
            assert len(tr) + len(va) == 30

    def test_kfold_invalid(self) -> None:
        from naviertwin.core.preprocessing.splitter import k_fold_indices

        with pytest.raises(ValueError):
            k_fold_indices(10, k=1)
