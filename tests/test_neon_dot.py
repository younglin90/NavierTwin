"""Round 535 — NEON dot proxy."""

from __future__ import annotations

import numpy as np


class TestNEON:
    def test_int8(self) -> None:
        from naviertwin.utils.neon_dot import dot_int8

        a = np.array([1, 2, 3], dtype=np.int8)
        b = np.array([4, 5, 6], dtype=np.int8)
        assert dot_int8(a, b) == 32

    def test_f32(self) -> None:
        from naviertwin.utils.neon_dot import dot_f32

        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        assert abs(dot_f32(a, b) - 11.0) < 1e-6
