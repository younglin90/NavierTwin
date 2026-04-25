"""Round 499 — gradient confusion."""

from __future__ import annotations

import numpy as np


class TestGC:
    def test_aligned_high(self) -> None:
        from naviertwin.utils.grad_confusion import grad_confusion

        # all gradients identical → ζ = 1
        g = np.tile(np.array([1.0, 0.0, 0.0]), (5, 1))
        assert abs(grad_confusion(g) - 1.0) < 1e-6

    def test_orthogonal_zero(self) -> None:
        from naviertwin.utils.grad_confusion import grad_confusion

        assert grad_confusion(np.eye(3)) == 0.0

    def test_opposite_negative(self) -> None:
        from naviertwin.utils.grad_confusion import grad_confusion

        g = np.array([[1.0, 0], [-1.0, 0]])
        assert abs(grad_confusion(g) + 1.0) < 1e-6
