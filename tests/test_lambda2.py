"""Round 169 — Λ₂."""

from __future__ import annotations

import numpy as np
import pytest


class TestLambda2:
    def test_shape(self) -> None:
        from naviertwin.core.analysis.lambda2 import lambda2_2d

        u = np.zeros((16, 16))
        v = np.zeros((16, 16))
        L = lambda2_2d(u, v)
        assert L.shape == (16, 16)

    def test_solid_rotation_negative(self) -> None:
        """u=-y, v=x → solid body rotation, Λ₂ < 0."""
        from naviertwin.core.analysis.lambda2 import lambda2_2d

        n = 32
        x = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, x, indexing="xy")
        u = -Y
        v = X
        L = lambda2_2d(u, v, x[1] - x[0], x[1] - x[0])
        # 내부 평균이 음수
        inner = L[5:-5, 5:-5]
        assert inner.mean() < 0

    def test_shear_non_vortex(self) -> None:
        """단순 shear 는 vortex 아님 → Λ₂ 평균이 0 근처."""
        from naviertwin.core.analysis.lambda2 import lambda2_2d

        n = 32
        y = np.linspace(0, 1, n)
        Y, _ = np.meshgrid(y, y, indexing="ij")
        u = Y  # linear shear u=y, v=0
        v = np.zeros_like(Y)
        L = lambda2_2d(u, v)
        # rotation 케이스보다 덜 음수
        assert abs(L[5:-5, 5:-5].mean()) < 1.0

    def test_numpy_fallback_matches_native_for_noncontiguous_float32(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import naviertwin.core.analysis.lambda2 as l2
        from naviertwin._native import HAS_NATIVE_KERNELS

        rng = np.random.default_rng(3)
        u = rng.standard_normal((20, 18)).astype(np.float32)[::2, ::2]
        v = rng.standard_normal((20, 18)).astype(np.float32)[::2, ::2]
        expected = l2._lambda2_2d_numpy(u, v, dx=0.2, dy=0.4)

        monkeypatch.setattr(l2, "_kernels", None)
        fallback = l2.lambda2_2d(u, v, dx=0.2, dy=0.4)
        np.testing.assert_allclose(fallback, expected)

        if HAS_NATIVE_KERNELS:
            monkeypatch.undo()
            native = l2.lambda2_2d(u, v, dx=0.2, dy=0.4)
            np.testing.assert_allclose(native, expected, rtol=1e-12, atol=1e-12)

    def test_numpy_fallback_input_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import naviertwin.core.analysis.lambda2 as l2

        monkeypatch.setattr(l2, "_kernels", None)
        with pytest.raises(ValueError, match="same shape"):
            l2.lambda2_2d(np.zeros((3, 4)), np.zeros((3, 5)))
        with pytest.raises(ValueError, match="2D"):
            l2.lambda2_2d(np.zeros((3, 4, 1)), np.zeros((3, 4, 1)))
        with pytest.raises(ValueError, match="non-zero"):
            l2.lambda2_2d(np.zeros((3, 4)), np.zeros((3, 4)), dy=0.0)
