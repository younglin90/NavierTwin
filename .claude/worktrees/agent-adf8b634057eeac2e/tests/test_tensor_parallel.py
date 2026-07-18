"""Round 415 — tensor parallel."""

from __future__ import annotations

import numpy as np


class TestTP:
    def test_column(self) -> None:
        from naviertwin.utils.tensor_parallel import column_parallel

        W = np.arange(12).reshape(3, 4)
        sh = column_parallel(W, 2)
        assert sh[0].shape == (3, 2)
        assert np.allclose(np.hstack(sh), W)

    def test_row(self) -> None:
        from naviertwin.utils.tensor_parallel import row_parallel

        W = np.arange(12).reshape(4, 3)
        sh = row_parallel(W, 2)
        assert sh[0].shape == (2, 3)
        assert np.allclose(np.vstack(sh), W)

    def test_all_reduce(self) -> None:
        from naviertwin.utils.tensor_parallel import all_reduce_sum

        a = [np.ones(3), np.ones(3) * 2]
        s = all_reduce_sum(a)
        assert np.allclose(s, 3.0)
