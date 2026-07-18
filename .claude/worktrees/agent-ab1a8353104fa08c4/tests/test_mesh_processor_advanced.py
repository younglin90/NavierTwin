"""Round 58 — pymeshlab advanced filters."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pymeshlab", reason="pymeshlab 필요")
pytest.importorskip("pyvista", reason="pyvista 필요")


def _sphere_mesh() -> object:
    import pyvista as pv

    return pv.Sphere(radius=1.0, theta_resolution=15, phi_resolution=15)


class TestAdvancedFilters:
    def test_taubin_smooth_preserves_bounds(self) -> None:
        from naviertwin.core.tools.mesh_processor_advanced import taubin_smooth

        sphere = _sphere_mesh()
        smoothed = taubin_smooth(sphere, lambda_=0.5, mu=-0.53, iterations=5)
        assert smoothed.n_points > 0
        # 구의 평균 반경 유지
        pts = np.asarray(smoothed.points)
        r = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
        assert 0.5 < float(r.mean()) < 2.0

    def test_fill_holes(self) -> None:
        from naviertwin.core.tools.mesh_processor_advanced import fill_holes

        sphere = _sphere_mesh()
        out = fill_holes(sphere, max_hole_size=10)
        assert out.n_points > 0

    def test_vertex_curvature_gaussian(self) -> None:
        from naviertwin.core.tools.mesh_processor_advanced import vertex_curvature

        sphere = _sphere_mesh()
        k = vertex_curvature(sphere)
        assert k.shape[0] > 0
        # 단위 구의 Gaussian curvature = 1 (어딘가 대략)
        assert np.all(np.isfinite(k))

    def test_isotropic_remesh(self) -> None:
        from naviertwin.core.tools.mesh_processor_advanced import isotropic_remesh

        sphere = _sphere_mesh()
        out = isotropic_remesh(sphere, target_edge_length=0.2, iterations=2)
        assert out.n_points > 0
