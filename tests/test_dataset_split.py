"""Round 515 — stratified split."""

from __future__ import annotations

import numpy as np


class TestSplit:
    def test_partition(self) -> None:
        from naviertwin.utils.dataset_split import stratified_split

        y = np.array([0] * 80 + [1] * 20)
        tr, va, te = stratified_split(y, ratios=(0.6, 0.2, 0.2), seed=0)
        # disjoint and cover all
        assert set(tr).isdisjoint(va)
        assert set(va).isdisjoint(te)
        assert len(tr) + len(va) + len(te) == 100

    def test_class_balance_preserved(self) -> None:
        from naviertwin.utils.dataset_split import stratified_split

        y = np.array([0] * 80 + [1] * 20)
        tr, va, te = stratified_split(y, ratios=(0.6, 0.2, 0.2))
        # train: 60% of 80 = 48; 60% of 20 = 12
        assert (y[tr] == 0).sum() == 48
        assert (y[tr] == 1).sum() == 12
