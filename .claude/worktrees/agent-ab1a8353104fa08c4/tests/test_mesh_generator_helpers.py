"""Round 581 — coverage uplift for tools.mesh_generator helpers (was 11%).

Exercises the gmsh-missing fallback path and the _msh_to_pv_ug converter
without requiring gmsh itself. The full mesh-generation paths are covered
in optional integration tests that pytest.importorskip('gmsh').
"""

from __future__ import annotations

import builtins

import pytest


class TestMeshGenerator:
    def test_require_gmsh_raises_helpful_message(self, monkeypatch) -> None:
        from naviertwin.core.tools import mesh_generator

        real_import = builtins.__import__

        def block_gmsh(name, *a, **kw):
            if name == "gmsh":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block_gmsh)
        with pytest.raises(RuntimeError, match="gmsh"):
            mesh_generator._require_gmsh()

    def test_msh_to_pv_ug_via_meshio(self, tmp_path) -> None:
        meshio = pytest.importorskip("meshio")
        pv = pytest.importorskip("pyvista")
        # write a tiny triangle .msh via meshio
        import numpy as np

        from naviertwin.core.tools.mesh_generator import _msh_to_pv_ug
        mesh = meshio.Mesh(
            points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
            cells=[("triangle", np.array([[0, 1, 2]]))],
        )
        path = tmp_path / "tri.msh"
        meshio.write(str(path), mesh, file_format="gmsh")
        ug = _msh_to_pv_ug(path)
        assert isinstance(ug, pv.UnstructuredGrid) or hasattr(ug, "points")
        # at least the 3 points should be present
        assert ug.n_points >= 3

    def test_msh_to_pv_ug_via_vtk(self, tmp_path) -> None:
        pv = pytest.importorskip("pyvista")
        from naviertwin.core.tools.mesh_generator import _msh_to_pv_ug

        # build a tiny grid (modern PyVista uses ImageData)
        grid_cls = getattr(pv, "ImageData", None) or getattr(pv, "UniformGrid")
        path = tmp_path / "x.vtu"
        grid_cls(dimensions=(3, 3, 1)).cast_to_unstructured_grid().save(str(path))
        out = _msh_to_pv_ug(path)
        assert hasattr(out, "n_points")
