"""Round 95 — 전역 seed."""

from __future__ import annotations

import random

import numpy as np


class TestSeeding:
    def test_python_numpy(self) -> None:
        from naviertwin.utils.seeding import set_global_seed

        set_global_seed(42)
        a = [random.random() for _ in range(3)]
        b = np.random.rand(3)
        set_global_seed(42)
        a2 = [random.random() for _ in range(3)]
        b2 = np.random.rand(3)
        assert a == a2
        assert np.allclose(b, b2)
