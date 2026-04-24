"""Round 86 — sliding window / IO pairs."""

from __future__ import annotations

import numpy as np
import pytest


class TestSlidingWindow:
    def test_basic_shape(self) -> None:
        from naviertwin.core.preprocessing.sliding_window import make_windows

        X = np.arange(20).reshape(1, 20).astype(np.float64)
        W = make_windows(X, window=4, stride=2)
        assert W.shape == (1, 9, 4)
        # first window = [0,1,2,3]
        assert list(W[0, 0]) == [0, 1, 2, 3]
        # second window = [2,3,4,5]
        assert list(W[0, 1]) == [2, 3, 4, 5]

    def test_stride_1(self) -> None:
        from naviertwin.core.preprocessing.sliding_window import make_windows

        X = np.arange(10).reshape(1, 10).astype(np.float64)
        W = make_windows(X, window=3, stride=1)
        assert W.shape == (1, 8, 3)

    def test_io_pairs(self) -> None:
        from naviertwin.core.preprocessing.sliding_window import make_io_pairs

        X = np.arange(30).reshape(2, 15).astype(np.float64)
        inp, tgt = make_io_pairs(X, in_len=4, out_len=2, stride=1)
        assert inp.shape[2] == 4
        assert tgt.shape[2] == 2
        assert inp.shape[:2] == tgt.shape[:2]

    def test_invalid(self) -> None:
        from naviertwin.core.preprocessing.sliding_window import make_windows

        with pytest.raises(ValueError):
            make_windows(np.zeros((2, 5)), window=10)
        with pytest.raises(ValueError):
            make_windows(np.zeros((2, 5)), window=0)
