"""Round 586 — Q-criterion + lambda2 happy-path coverage (was 45%)."""

from __future__ import annotations

import numpy as np
import pytest


def _make_simple_mesh():
    pv = pytest.importorskip("pyvista")
    # tiny structured grid with vortex-like velocity (rotation about z)
    grid = pv.ImageData(dimensions=(8, 8, 8), spacing=(0.1, 0.1, 0.1))
    pts = grid.points
    # u = (-y, x, 0) → solid-body rotation
    U = np.zeros_like(pts)
    U[:, 0] = -pts[:, 1]
    U[:, 1] = pts[:, 0]
    grid.point_data["U"] = U
    return grid.cast_to_unstructured_grid()


class TestQLambda:
    def test_compute_q_pyvista_path(self) -> None:
        pv = pytest.importorskip("pyvista")
        from naviertwin.core.flow_analysis.vortex.q_criterion import (
            compute_q_criterion,
        )

        mesh = _make_simple_mesh()
        out = compute_q_criterion(mesh, velocity_name="U")
        assert "Q-criterion" in out.point_data
        # solid-body rotation → Omega dominates → Q > 0
        q = np.asarray(out.point_data["Q-criterion"])
        assert (q > 0).any()
        assert "vorticity" in out.point_data
        _ = pv

    def test_compute_lambda2(self) -> None:
        pytest.importorskip("pyvista")
        from naviertwin.core.flow_analysis.vortex.q_criterion import compute_lambda2

        mesh = _make_simple_mesh()
        out = compute_lambda2(mesh, velocity_name="U")
        assert "lambda2" in out.point_data
        l2 = np.asarray(out.point_data["lambda2"])
        # rotation core → λ₂ negative
        assert (l2 < 0).any()

    def test_gradient_numpy_helpers_match_native(self) -> None:
        from naviertwin._native import HAS_NATIVE_KERNELS, _kernels
        from naviertwin.core.flow_analysis.vortex.q_criterion import (
            _compute_lambda2_from_gradient_numpy,
            _compute_q_from_gradient_numpy,
        )

        rng = np.random.default_rng(8)
        grad = rng.standard_normal((18, 3, 3)).astype(np.float32)[::2]
        flat_grad = np.ascontiguousarray(grad).reshape(-1, 9)

        q_tensor, vort_tensor = _compute_q_from_gradient_numpy(grad)
        q_flat, vort_flat = _compute_q_from_gradient_numpy(flat_grad)
        l2_tensor = _compute_lambda2_from_gradient_numpy(grad)
        l2_flat = _compute_lambda2_from_gradient_numpy(flat_grad)
        np.testing.assert_allclose(q_tensor, q_flat)
        np.testing.assert_allclose(vort_tensor, vort_flat)
        np.testing.assert_allclose(l2_tensor, l2_flat)

        if HAS_NATIVE_KERNELS:
            native_q, native_vort = _kernels.q_criterion_from_grad_3d(grad)
            native_l2 = _kernels.lambda2_from_grad_3d(grad)
            np.testing.assert_allclose(native_q, q_tensor, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(native_vort, vort_tensor, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(native_l2, l2_tensor, rtol=1e-10, atol=1e-10)

    def test_gradient_numpy_helpers_reject_bad_shape(self) -> None:
        from naviertwin.core.flow_analysis.vortex.q_criterion import (
            _compute_lambda2_from_gradient_numpy,
            _compute_q_from_gradient_numpy,
        )

        with pytest.raises(ValueError, match="gradient must have shape"):
            _compute_q_from_gradient_numpy(np.zeros((4, 8)))
        with pytest.raises(ValueError, match="gradient must have shape"):
            _compute_lambda2_from_gradient_numpy(np.zeros((4, 3, 2)))
