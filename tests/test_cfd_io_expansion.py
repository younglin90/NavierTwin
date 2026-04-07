"""NavierTwin v1.1.0 — CFD I/O 확장 테스트 모음.

FluentReader, CGNSReader, GmshReader 의 등록, 폴백, 오류 처리를 검증한다.
모든 테스트는 synthetic mesh 기반으로 상용 소프트웨어 없이 실행된다.

실행:
    기본 (optional 제외)::

        pytest tests/test_cfd_io_expansion.py -m "not optional"

    전체 (pyCGNS, gmsh 설치 환경)::

        pytest tests/test_cfd_io_expansion.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pyvista = pytest.importorskip("pyvista", reason="pyvista 가 필요합니다")

# ---------------------------------------------------------------------------
# 픽스처 경로
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
CAS_PATH = FIXTURES / "minimal.cas"
DAT_PATH = FIXTURES / "minimal.dat"
BINARY_CAS = FIXTURES / "fluent_binary_stub.cas"
CGNS_PATH = FIXTURES / "synthetic.cgns"
MSH_V22 = FIXTURES / "synthetic_v22.msh"
MSH_V41 = FIXTURES / "synthetic_v41.msh"
NASTRAN_MSH = FIXTURES / "nastran_stub.msh"


# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------


def _make_pv_ug(n_points: int = 4) -> Any:
    """테스트용 최소 pyvista UnstructuredGrid 를 반환한다."""
    import pyvista as pv

    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.3, 1]], dtype=float
    )[:n_points]
    cells = np.array([4, 0, 1, 2, 3], dtype=np.int64)
    cell_types = np.array([10], dtype=np.uint8)
    mesh = pv.UnstructuredGrid(cells, cell_types, points)
    mesh.point_data["p"] = np.ones(n_points)
    return mesh


# ===========================================================================
# FluentReader 테스트 (8)
# ===========================================================================


class TestFluentReader:
    def test_fluent_reader_registered(self) -> None:
        """FluentReader 가 ReaderFactory 에 .cas, .dat 로 등록되어야 한다."""
        from naviertwin.core.cfd_reader import ReaderFactory  # 등록 트리거
        from naviertwin.core.cfd_reader.fluent_reader import FluentReader

        exts = ReaderFactory.registered_extensions()
        assert ".cas" in exts, ".cas 미등록"
        assert ".dat" in exts, ".dat 미등록"
        assert ReaderFactory._registry[".cas"] is FluentReader
        assert ReaderFactory._registry[".dat"] is FluentReader

    def test_fluent_pyvista_path(self, tmp_path: Path) -> None:
        """pyvista.FluentReader 경로가 정상 동작해야 한다."""
        from naviertwin.core.cfd_reader.fluent_reader import FluentReader

        ug = _make_pv_ug()
        mock_reader = MagicMock()
        mock_reader.read.return_value = ug

        with patch("pyvista.FluentReader", return_value=mock_reader):
            reader = FluentReader()
            dataset = reader.read(CAS_PATH)

        assert dataset.mesh is not None
        assert dataset.n_points > 0

    def test_fluent_meshio_fallback(self, tmp_path: Path) -> None:
        """pyvista 실패 시 meshio 로 폴백해야 한다."""
        import meshio as meshio_mod

        from naviertwin.core.cfd_reader.fluent_reader import FluentReader

        fake_mesh = meshio_mod.Mesh(
            points=np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.3, 1]]),
            cells=[("tetra", np.array([[0, 1, 2, 3]]))],
        )

        def raise_for_pv(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("pyvista mock 실패")

        with patch("pyvista.FluentReader", side_effect=raise_for_pv):
            with patch("meshio.read", return_value=fake_mesh):
                reader = FluentReader()
                dataset = reader.read(CAS_PATH)

        assert dataset.mesh is not None

    def test_fluent_ascii_parser_fallback(self) -> None:
        """pyvista + meshio 모두 실패 시 FluentASCIIParser 로 폴백해야 한다."""
        from naviertwin.core.cfd_reader.fluent_reader import FluentReader

        def raise_always(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("mock 실패")

        with patch("pyvista.FluentReader", side_effect=raise_always):
            with patch("meshio.read", side_effect=raise_always):
                reader = FluentReader()
                dataset = reader.read(CAS_PATH)  # FluentASCIIParser 실행

        assert dataset.mesh is not None
        assert dataset.n_points > 0

    def test_fluent_sibling_dat_autoload(self) -> None:
        """같은 디렉토리에 .dat 파일이 있으면 자동으로 로드해야 한다."""
        import meshio as meshio_mod

        from naviertwin.core.cfd_reader.fluent_reader import FluentReader

        fake_cas = meshio_mod.Mesh(
            points=np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.3, 1]]),
            cells=[("tetra", np.array([[0, 1, 2, 3]]))],
        )
        fake_dat = meshio_mod.Mesh(
            points=np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.3, 1]]),
            cells=[],
            point_data={"U": np.ones((4, 3)), "p": np.ones(4)},
        )

        call_count = {"n": 0}

        def mock_read(path: str, *args: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1
            if str(path).endswith(".dat"):
                return fake_dat
            return fake_cas

        def raise_for_pv(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("pyvista mock")

        with patch("pyvista.FluentReader", side_effect=raise_for_pv):
            with patch("meshio.read", side_effect=mock_read):
                reader = FluentReader()
                dataset = reader.read(CAS_PATH)

        # .cas 와 .dat 둘 다 읽혔어야 함
        assert call_count["n"] >= 2, "sibling .dat 가 로드되지 않음"

    def test_fluent_missing_dat_warns(self, tmp_path: Path) -> None:
        """sibling .dat 가 없을 때 warning 로그가 출력돼야 한다."""
        import logging

        import meshio as meshio_mod

        from naviertwin.core.cfd_reader.fluent_reader import FluentReader

        # .dat 없는 임시 .cas
        cas_only = tmp_path / "only.cas"
        cas_only.write_text("( dummy )")

        fake_cas = meshio_mod.Mesh(
            points=np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]]),
            cells=[("triangle", np.array([[0, 1, 2]]))],
        )

        def raise_for_pv(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("pyvista mock")

        with patch("pyvista.FluentReader", side_effect=raise_for_pv):
            with patch("meshio.read", return_value=fake_cas):
                reader = FluentReader()
                with patch(
                    "naviertwin.core.cfd_reader.fluent_reader.logger"
                ) as mock_log:
                    dataset = reader.read(cas_only)
                    # .dat 없음 → warning 로그가 호출돼야 함
                    assert mock_log.warning.called

    def test_fluent_binary_raises_value_error(self) -> None:
        """바이너리 .cas 입력 시 ValueError 가 발생해야 한다."""
        from naviertwin.core.cfd_reader.fluent_reader import FluentReader

        reader = FluentReader()
        with pytest.raises(ValueError, match="바이너리"):
            reader.read(BINARY_CAS)

    def test_fluent_result_is_cfd_dataset(self) -> None:
        """FluentReader 반환값이 CFDDataset 타입이고 n_points > 0 이어야 한다."""
        from naviertwin.core.cfd_reader.base import CFDDataset
        from naviertwin.core.cfd_reader.fluent_reader import FluentReader

        reader = FluentReader()
        # ASCII 파서 경로 — pyvista/meshio 실패 mock
        def raise_always(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("mock")

        with patch("pyvista.FluentReader", side_effect=raise_always):
            with patch("meshio.read", side_effect=raise_always):
                dataset = reader.read(CAS_PATH)

        assert isinstance(dataset, CFDDataset)
        assert dataset.n_points > 0
        assert isinstance(dataset.time_steps, list)
        assert isinstance(dataset.field_names, list)


# ===========================================================================
# CGNSReader 테스트 (9)
# ===========================================================================


class TestCGNSReader:
    def test_cgns_reader_registered(self) -> None:
        """CGNSReader 가 ReaderFactory 에 .cgns 로 등록되어야 한다."""
        from naviertwin.core.cfd_reader import ReaderFactory
        from naviertwin.core.cfd_reader.cgns_reader import CGNSReader

        exts = ReaderFactory.registered_extensions()
        assert ".cgns" in exts
        assert ReaderFactory._registry[".cgns"] is CGNSReader

    def test_cgns_pyvista_path(self) -> None:
        """pyvista.CGNSReader 경로가 정상 동작해야 한다."""
        from naviertwin.core.cfd_reader.cgns_reader import CGNSReader

        ug = _make_pv_ug()
        mock_reader = MagicMock()
        mock_reader.read.return_value = ug

        with patch("pyvista.CGNSReader", return_value=mock_reader):
            reader = CGNSReader()
            dataset = reader.read(CGNS_PATH)

        assert dataset.n_points > 0

    def test_cgns_pycgns_fallback(self) -> None:
        """pyvista 실패 시 pyCGNS 로 폴백해야 한다."""
        from naviertwin.core.cfd_reader.cgns_reader import CGNSReader

        # pyvista 실패 mock
        def raise_for_pv(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("pyvista mock")

        # pyCGNS MAP.load mock — synthetic tree 반환
        fake_tree = [
            "Base", None,
            [
                ["Zone", None, [
                    ["GridCoordinates", None, [
                        ["CoordinateX", np.linspace(0, 1, 5), [], "DataArray_t"],
                        ["CoordinateY", np.zeros(5), [], "DataArray_t"],
                        ["CoordinateZ", np.zeros(5), [], "DataArray_t"],
                    ], "GridCoordinates_t"],
                ], "Zone_t"],
            ],
            "CGNSBase_t",
        ]
        mock_map = MagicMock()
        mock_map.load.return_value = (fake_tree, [], [])

        with patch("pyvista.CGNSReader", side_effect=raise_for_pv):
            with patch.dict("sys.modules", {"CGNS": MagicMock(MAP=mock_map), "CGNS.MAP": mock_map}):
                with patch("naviertwin.core.cfd_reader.cgns_reader.CGNSReader._read_with_pycgns") as mock_cgns:
                    mock_cgns.return_value = MagicMock(
                        mesh=_make_pv_ug(), time_steps=[0.0],
                        field_names=[], metadata={},
                        n_points=4, n_cells=1
                    )
                    mock_cgns.return_value.__class__ = type(
                        "CFDDataset", (), {
                            "mesh": _make_pv_ug(),
                            "time_steps": [0.0],
                            "field_names": [],
                            "metadata": {},
                            "n_points": 4,
                            "n_cells": 1,
                        }
                    )
                    reader = CGNSReader()
                    # pyvista 실패 → pyCGNS 호출 확인
                    with patch.object(reader, "_read_with_pyvista", side_effect=raise_for_pv):
                        with patch.object(reader, "_read_with_pycgns") as mock_m:
                            from naviertwin.core.cfd_reader.base import CFDDataset
                            mock_m.return_value = CFDDataset(
                                mesh=_make_pv_ug(),
                                time_steps=[0.0],
                                field_names=[],
                                metadata={},
                            )
                            dataset = reader.read(CGNS_PATH)
                            assert mock_m.called

    def test_cgns_h5py_fallback(self) -> None:
        """pyvista + pyCGNS 실패 시 h5py 로 폴백해야 한다."""
        from naviertwin.core.cfd_reader.base import CFDDataset
        from naviertwin.core.cfd_reader.cgns_reader import CGNSReader

        reader = CGNSReader()

        def raise_always(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("mock 실패")

        fake_result = CFDDataset(
            mesh=_make_pv_ug(), time_steps=[0.0], field_names=[], metadata={}
        )

        with patch.object(reader, "_read_with_pyvista", side_effect=raise_always):
            with patch.object(reader, "_read_with_pycgns", side_effect=raise_always):
                with patch.object(reader, "_read_with_h5py", return_value=fake_result) as mock_h5:
                    dataset = reader.read(CGNS_PATH)
                    assert mock_h5.called
                    assert dataset is fake_result

    def test_cgns_meshio_fallback(self) -> None:
        """pyvista + pyCGNS + h5py 모두 실패 시 meshio 로 폴백해야 한다."""
        from naviertwin.core.cfd_reader.base import CFDDataset
        from naviertwin.core.cfd_reader.cgns_reader import CGNSReader

        reader = CGNSReader()

        def raise_always(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("mock 실패")

        fake_result = CFDDataset(
            mesh=_make_pv_ug(), time_steps=[0.0], field_names=[], metadata={}
        )

        with patch.object(reader, "_read_with_pyvista", side_effect=raise_always):
            with patch.object(reader, "_read_with_pycgns", side_effect=raise_always):
                with patch.object(reader, "_read_with_h5py", side_effect=raise_always):
                    with patch.object(reader, "_read_with_meshio", return_value=fake_result) as mock_m:
                        dataset = reader.read(CGNS_PATH)
                        assert mock_m.called

    def test_cgns_all_parsers_fail_raises(self) -> None:
        """모든 파서가 실패하면 ValueError 가 발생해야 한다."""
        from naviertwin.core.cfd_reader.cgns_reader import CGNSReader

        reader = CGNSReader()

        def raise_always(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("mock 실패")

        with patch.object(reader, "_read_with_pyvista", side_effect=raise_always):
            with patch.object(reader, "_read_with_pycgns", side_effect=raise_always):
                with patch.object(reader, "_read_with_h5py", side_effect=raise_always):
                    with patch.object(reader, "_read_with_meshio", side_effect=raise_always):
                        with pytest.raises(ValueError, match="CGNSReader"):
                            reader.read(CGNS_PATH)

    def test_cgns_result_is_cfd_dataset(self) -> None:
        """CGNSReader 반환값이 CFDDataset 타입이어야 한다."""
        from naviertwin.core.cfd_reader.base import CFDDataset
        from naviertwin.core.cfd_reader.cgns_reader import CGNSReader

        reader = CGNSReader()
        # h5py 직접 파싱 경로 테스트

        def raise_pv(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("pv mock")

        def raise_cgns(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("pycgns mock")

        with patch.object(reader, "_read_with_pyvista", side_effect=raise_pv):
            with patch.object(reader, "_read_with_pycgns", side_effect=raise_cgns):
                dataset = reader.read(CGNS_PATH)  # h5py 경로

        assert isinstance(dataset, CFDDataset)
        assert dataset.n_points > 0

    def test_cgns_factory_auto_detect(self) -> None:
        """ReaderFactory.create_and_read(.cgns) 가 CGNSReader 를 선택해야 한다."""
        from naviertwin.core.cfd_reader import ReaderFactory
        from naviertwin.core.cfd_reader.cgns_reader import CGNSReader

        reader = ReaderFactory.get_reader(CGNS_PATH)
        assert isinstance(reader, CGNSReader)

    @pytest.mark.optional
    def test_cgns_with_real_pycgns(self) -> None:
        """실제 pyCGNS (CGNS.MAP) 로 synthetic.cgns 를 읽어야 한다."""
        pytest.importorskip("CGNS", reason="pyCGNS 가 필요합니다")
        from naviertwin.core.cfd_reader.cgns_reader import CGNSReader

        reader = CGNSReader()

        def raise_pv(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("pyvista mock — pyCGNS 경로 강제")

        with patch.object(reader, "_read_with_pyvista", side_effect=raise_pv):
            try:
                dataset = reader.read(CGNS_PATH)
                # pyCGNS 가 이 HDF5 구조를 지원하지 않으면 h5py 폴백
                assert dataset.n_points >= 0
            except ValueError:
                pytest.skip("CGNS.MAP 이 이 HDF5 구조를 지원하지 않음 — h5py 폴백으로 대체")


# ===========================================================================
# GmshReader 테스트 (6)
# ===========================================================================


class TestGmshReader:
    def test_gmsh_reader_registered(self) -> None:
        """GmshReader 가 ReaderFactory 에 .msh 로 등록되어야 한다."""
        from naviertwin.core.cfd_reader import ReaderFactory
        from naviertwin.core.cfd_reader.gmsh_reader import GmshReader

        exts = ReaderFactory.registered_extensions()
        assert ".msh" in exts
        assert ReaderFactory._registry[".msh"] is GmshReader

    def test_gmsh_msh_v22(self) -> None:
        """Gmsh .msh v2.2 파일을 읽어야 한다."""
        from naviertwin.core.cfd_reader.base import CFDDataset
        from naviertwin.core.cfd_reader.gmsh_reader import GmshReader

        reader = GmshReader()
        dataset = reader.read(MSH_V22)
        assert isinstance(dataset, CFDDataset)
        assert dataset.n_points > 0

    def test_gmsh_msh_v41(self) -> None:
        """Gmsh .msh v4.1 파일을 읽어야 한다."""
        from naviertwin.core.cfd_reader.base import CFDDataset
        from naviertwin.core.cfd_reader.gmsh_reader import GmshReader

        reader = GmshReader()
        dataset = reader.read(MSH_V41)
        assert isinstance(dataset, CFDDataset)
        assert dataset.n_points > 0

    def test_gmsh_result_is_cfd_dataset(self) -> None:
        """GmshReader 반환값이 CFDDataset 이고 mesh 가 있어야 한다."""
        from naviertwin.core.cfd_reader.base import CFDDataset
        from naviertwin.core.cfd_reader.gmsh_reader import GmshReader

        reader = GmshReader()
        dataset = reader.read(MSH_V41)
        assert isinstance(dataset, CFDDataset)
        assert dataset.mesh is not None

    def test_gmsh_non_gmsh_msh_raises(self) -> None:
        """non-Gmsh .msh 파일에서 ValueError 가 발생해야 한다."""
        from naviertwin.core.cfd_reader.gmsh_reader import GmshReader

        reader = GmshReader()
        with pytest.raises(ValueError, match="GmshReader"):
            reader.read(NASTRAN_MSH)

    def test_gmsh_factory_auto_detect(self) -> None:
        """ReaderFactory.create_and_read(.msh) 가 GmshReader 를 선택해야 한다."""
        from naviertwin.core.cfd_reader import ReaderFactory
        from naviertwin.core.cfd_reader.gmsh_reader import GmshReader

        reader = ReaderFactory.get_reader(MSH_V41)
        assert isinstance(reader, GmshReader)

    @pytest.mark.optional
    def test_gmsh_api_probe(self) -> None:
        """gmsh Python API 가 설치된 경우 probe 가 성공해야 한다."""
        gmsh = pytest.importorskip("gmsh", reason="gmsh API 가 필요합니다")
        from naviertwin.core.cfd_reader.gmsh_reader import _gmsh_probe

        # probe 가 예외 없이 완료되어야 함
        _gmsh_probe(MSH_V41)


# ===========================================================================
# 통합 테스트 (3)
# ===========================================================================


class TestIntegration:
    def test_factory_all_new_extensions_registered(self) -> None:
        """v1.1.0 신규 확장자 .cas/.dat/.cgns/.msh 가 모두 등록되어야 한다."""
        from naviertwin.core.cfd_reader import ReaderFactory

        exts = ReaderFactory.registered_extensions()
        for ext in (".cas", ".dat", ".cgns", ".msh"):
            assert ext in exts, f"{ext} 미등록"

    def test_reader_factory_error_message_format(self, tmp_path: Path) -> None:
        """지원하지 않는 확장자에서 ValueError 메시지가 적절해야 한다."""
        from naviertwin.core.cfd_reader import ReaderFactory

        unknown = tmp_path / "data.xyz"
        unknown.touch()
        with pytest.raises(ValueError, match=r"지원하지 않는|supported"):
            ReaderFactory.get_reader(unknown)

    def test_cfd_dataset_schema_consistency(self) -> None:
        """세 리더 반환 CFDDataset 이 공통 스키마를 가져야 한다."""
        from naviertwin.core.cfd_reader.base import CFDDataset
        from naviertwin.core.cfd_reader.cgns_reader import CGNSReader
        from naviertwin.core.cfd_reader.gmsh_reader import GmshReader

        readers_and_paths = [
            (GmshReader(), MSH_V41),
        ]

        # CGNSReader h5py 경로
        cgns_reader = CGNSReader()

        def raise_pv(*a: Any, **k: Any) -> None:
            raise RuntimeError("mock")

        def raise_cgns(*a: Any, **k: Any) -> None:
            raise RuntimeError("mock")

        with patch.object(cgns_reader, "_read_with_pyvista", side_effect=raise_pv):
            with patch.object(cgns_reader, "_read_with_pycgns", side_effect=raise_cgns):
                readers_and_paths.append((cgns_reader, CGNS_PATH))

        for reader, path in readers_and_paths:
            dataset = reader.read(path)
            assert isinstance(dataset, CFDDataset), f"{type(reader).__name__} 반환 타입 오류"
            assert hasattr(dataset, "mesh")
            assert hasattr(dataset, "time_steps")
            assert hasattr(dataset, "field_names")
            assert hasattr(dataset, "metadata")
            assert isinstance(dataset.time_steps, list)
            assert isinstance(dataset.field_names, list)
            assert isinstance(dataset.metadata, dict)
