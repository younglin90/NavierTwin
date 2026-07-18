"""Round 204 — velocity gradient decomposition."""

from __future__ import annotations

import numpy as np
import pytest


class TestVG:
    def test_decompose(self) -> None:
        from naviertwin.core.analysis.velocity_gradient import decompose_J_3x3

        rng = np.random.default_rng(0)
        J = rng.standard_normal((3, 3))
        S, W = decompose_J_3x3(J)
        assert np.allclose(S, S.T)
        assert np.allclose(W, -W.T)
        assert np.allclose(S + W, J)

    def test_invariants_identity(self) -> None:
        from naviertwin.core.analysis.velocity_gradient import invariants_3x3

        J = np.eye(3)
        inv = invariants_3x3(J)
        assert inv["P"] == -3.0
        assert inv["R"] == -1.0

    def test_invalid(self) -> None:
        from naviertwin.core.analysis.velocity_gradient import decompose_J_3x3

        with pytest.raises(ValueError):
            decompose_J_3x3(np.eye(4))

    def test_field_j_2d(self) -> None:
        from naviertwin.core.analysis.velocity_gradient import field_J_2d

        rng = np.random.default_rng(0)
        u = rng.standard_normal((10, 12))
        v = rng.standard_normal((10, 12))
        J = field_J_2d(u, v)
        assert J.shape == (10, 12, 2, 2)

    def test_numpy_fallback_matches_native_for_2d_noncontiguous(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import naviertwin.core.analysis.velocity_gradient as vg
        from naviertwin._native import HAS_NATIVE_KERNELS

        rng = np.random.default_rng(1)
        u = rng.standard_normal((18, 20)).astype(np.float32)[::2, ::2]
        v = rng.standard_normal((18, 20)).astype(np.float32)[::2, ::2]
        fallback = vg._field_J_2d_numpy(u, v, dx=0.25, dy=0.5)

        monkeypatch.setattr(vg, "_kernels", None)
        public_fallback = vg.field_J_2d(u, v, dx=0.25, dy=0.5)
        np.testing.assert_allclose(public_fallback, fallback)

        if HAS_NATIVE_KERNELS:
            monkeypatch.undo()
            native = vg.field_J_2d(u, v, dx=0.25, dy=0.5)
            np.testing.assert_allclose(native, fallback, rtol=1e-12, atol=1e-12)

    def test_numpy_fallback_matches_native_for_3x3(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import naviertwin.core.analysis.velocity_gradient as vg
        from naviertwin._native import HAS_NATIVE_KERNELS

        J = np.array([[1.0, 2.0, -0.5], [0.25, -1.0, 3.0], [4.0, -2.0, 0.5]])
        expected_s, expected_w = vg._decompose_J_3x3_numpy(J)
        expected_inv = vg._invariants_3x3_numpy(J)

        monkeypatch.setattr(vg, "_kernels", None)
        fallback_s, fallback_w = vg.decompose_J_3x3(J.astype(np.float32))
        fallback_inv = vg.invariants_3x3(J.astype(np.float32))
        np.testing.assert_allclose(fallback_s, expected_s)
        np.testing.assert_allclose(fallback_w, expected_w)
        assert fallback_inv == pytest.approx(expected_inv)

        if HAS_NATIVE_KERNELS:
            monkeypatch.undo()
            native_s, native_w = vg.decompose_J_3x3(J)
            native_inv = vg.invariants_3x3(J)
            np.testing.assert_allclose(native_s, expected_s)
            np.testing.assert_allclose(native_w, expected_w)
            assert native_inv == pytest.approx(expected_inv)

    def test_numpy_fallback_input_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import naviertwin.core.analysis.velocity_gradient as vg

        monkeypatch.setattr(vg, "_kernels", None)
        with pytest.raises(ValueError, match="3x3"):
            vg.decompose_J_3x3(np.eye(4))
        with pytest.raises(ValueError, match="same shape"):
            vg.field_J_2d(np.zeros((3, 4)), np.zeros((3, 5)))
        with pytest.raises(ValueError, match="at least 2"):
            vg.field_J_2d(np.zeros((1, 4)), np.zeros((1, 4)))
        with pytest.raises(ValueError, match="non-zero"):
            vg.field_J_2d(np.zeros((3, 4)), np.zeros((3, 4)), dx=0.0)
