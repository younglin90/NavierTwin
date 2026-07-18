"""Round 285 — C4 equivariant CNN."""

from __future__ import annotations

import pytest


class TestEquivariant:
    def test_forward_shape(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.neural.equivariant_cnn import C4Conv2d

        conv = C4Conv2d(2, 4, kernel_size=3)
        x = torch.randn(1, 2, 16, 16)
        y = conv(x)
        assert y.shape == (1, 4, 16, 16)

    def test_rotation_equivariance(self) -> None:
        """Output of conv(rot90(x)) ≈ rot90(conv(x)) (group-avg → invariant under rotation)."""
        torch = pytest.importorskip("torch")
        from naviertwin.core.neural.equivariant_cnn import C4Conv2d

        torch.manual_seed(0)
        conv = C4Conv2d(1, 1, kernel_size=3)
        x = torch.randn(1, 1, 8, 8)
        y1 = conv(x)
        y2 = conv(torch.rot90(x, 1, dims=(-2, -1)))
        # group-averaged conv → rot(y1) == y2 (up to rotation alignment)
        assert torch.allclose(torch.rot90(y1, 1, dims=(-2, -1)), y2, atol=1e-5)
