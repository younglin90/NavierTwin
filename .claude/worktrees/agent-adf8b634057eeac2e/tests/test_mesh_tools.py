"""mesh_generator / mesh_processor 테스트 모음.

quality_report 는 core 의존성만으로 통과해야 한다.
generate_* / simplify / smooth 는 optional 의존성 필요 시 자동 스킵.
"""

from __future__ import annotations

import numpy as np
import pytest

pyvista = pytest.importorskip("pyvista", reason="pyvista 가 필요합니다")


def _simple_tetra_ug() -> object:
    import pyvista as pv

    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
    )
    cells = np.array([4, 0, 1, 2, 3], dtype=np.int64)
    cell_types = np.array([10], dtype=np.uint8)
    return pv.UnstructuredGrid(cells, cell_types, points)


class TestQualityReport:
    def test_quality_report_core_deps_only(self) -> None:
        """quality_report 는 optional 의존성 없이 동작해야 한다."""
        from naviertwin.core.tools.mesh_processor import quality_report

        mesh = _simple_tetra_ug()
        rep = quality_report(mesh)
        assert rep["n_points"] == 4
        assert rep["n_cells"] == 1
        # aspect_ratio 는 VTK 버전에 따라 다를 수 있어 존재 여부만 확인
        assert any(
            k.startswith("aspect_ratio") or k.startswith("scaled_jacobian")
            for k in rep
        )


class TestMeshGeneratorOptional:
    @pytest.mark.optional
    def test_generate_channel(self) -> None:
        gmsh = pytest.importorskip("gmsh", reason="gmsh 가 필요합니다")
        del gmsh
        from naviertwin.core.tools.mesh_generator import generate_channel

        mesh = generate_channel(length=2.0, height=1.0, nx=10, ny=5)
        assert mesh.n_points > 0
        assert mesh.n_cells > 0

    @pytest.mark.optional
    def test_generate_airfoil(self) -> None:
        pytest.importorskip("gmsh", reason="gmsh 가 필요합니다")
        from naviertwin.core.tools.mesh_generator import generate_airfoil

        mesh = generate_airfoil(naca_code="0012", chord=1.0, n_points=40)
        assert mesh.n_points > 10

    def test_missing_gmsh_error_message(self) -> None:
        """gmsh 미설치 시 친절한 RuntimeError 메시지가 나와야 한다."""
        from unittest.mock import patch

        from naviertwin.core.tools import mesh_generator

        with patch.object(
            mesh_generator, "_require_gmsh", side_effect=RuntimeError("gmsh 설치 필요")
        ):
            with pytest.raises(RuntimeError, match="gmsh"):
                mesh_generator.generate_channel(1.0, 1.0, 5, 5)


class TestMeshProcessorOptional:
    @pytest.mark.optional
    def test_simplify_requires_pymeshlab(self) -> None:
        try:
            import pymeshlab  # noqa: F401
        except ImportError:
            pytest.skip("pymeshlab 가 필요합니다")
        from naviertwin.core.tools.mesh_processor import simplify

        mesh = _simple_tetra_ug()
        out = simplify(mesh, target_faces=2)
        assert out.n_points > 0

    def test_simplify_without_pymeshlab_raises(self) -> None:
        """pymeshlab 미설치 시 RuntimeError 가 발생해야 한다."""
        from unittest.mock import patch

        from naviertwin.core.tools import mesh_processor

        def _raise(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("pymeshlab 설치 필요")

        with patch.object(mesh_processor, "_to_pymeshlab_mesh", side_effect=_raise):
            with pytest.raises(RuntimeError, match="pymeshlab"):
                mesh_processor.simplify(_simple_tetra_ug(), 100)
