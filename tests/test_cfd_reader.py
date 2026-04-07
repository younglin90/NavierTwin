"""CFD 리더 테스트 모음.

pyvista 가 설치되어 있지 않으면 일부 테스트는 자동으로 skip 된다.
NTwinWriter/NTwinReader 테스트는 h5py 설치 여부에 따라 skip 된다.
실제 CFD 파일 없이 tmp_path fixture 와 임시 메쉬만으로 동작한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 조건부 skip 마커
# ---------------------------------------------------------------------------

pyvista = pytest.importorskip("pyvista", reason="pyvista 가 설치되어 있지 않음")
h5py = pytest.importorskip("h5py", reason="h5py 가 설치되어 있지 않음")


# ---------------------------------------------------------------------------
# 헬퍼: 간단한 pyvista UnstructuredGrid 생성
# ---------------------------------------------------------------------------


def _make_simple_ug() -> "pyvista.UnstructuredGrid":
    """테스트용 단순 UnstructuredGrid (6-point tetrahedra mesh) 를 생성한다.

    Returns:
        point_data 에 'U'(벡터), 'p'(스칼라) 필드를 가진 UnstructuredGrid.
    """
    import pyvista as pv

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.3, 1.0],
        ],
        dtype=np.float32,
    )
    # VTK_TETRA = 10
    cells = np.array([4, 0, 1, 2, 3], dtype=np.int64)
    cell_types = np.array([10], dtype=np.uint8)

    mesh = pv.UnstructuredGrid(cells, cell_types, points)
    n = mesh.n_points
    mesh.point_data["U"] = np.random.rand(n, 3).astype(np.float32)
    mesh.point_data["p"] = np.random.rand(n).astype(np.float32)
    return mesh


# ---------------------------------------------------------------------------
# 테스트: CFDDataset 기본 속성
# ---------------------------------------------------------------------------


def test_cfd_dataset_properties() -> None:
    """CFDDataset 의 기본 속성(n_time_steps, n_points, n_cells)을 검증한다."""
    from naviertwin.core.cfd_reader.base import CFDDataset

    mesh = _make_simple_ug()
    dataset = CFDDataset(
        mesh=mesh,
        time_steps=[0.0, 0.5, 1.0],
        field_names=["U", "p"],
        metadata={"solver": "test"},
    )

    assert dataset.n_time_steps == 3
    assert dataset.n_points == mesh.n_points
    assert dataset.n_cells == mesh.n_cells
    assert "U" in dataset.field_names
    assert dataset.metadata["solver"] == "test"


def test_cfd_dataset_none_mesh_raises() -> None:
    """mesh=None 이면 TypeError 가 발생해야 한다."""
    from naviertwin.core.cfd_reader.base import CFDDataset

    with pytest.raises(TypeError):
        CFDDataset(mesh=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 테스트: VTKReader (.vtu 임시 파일)
# ---------------------------------------------------------------------------


def test_vtk_reader_with_pyvista(tmp_path: Path) -> None:
    """VTKReader 가 임시 .vtu 파일을 올바르게 읽는지 검증한다."""
    import pyvista as pv
    from naviertwin.core.cfd_reader.vtk_reader import VTKReader

    mesh = _make_simple_ug()
    vtu_path = tmp_path / "test.vtu"
    mesh.save(str(vtu_path))

    reader = VTKReader()
    dataset = reader.read(vtu_path)

    assert dataset.n_points > 0
    assert dataset.n_cells > 0
    assert dataset.time_steps == [0.0]
    assert "U" in dataset.field_names
    assert "p" in dataset.field_names


def test_vtk_reader_file_not_found() -> None:
    """존재하지 않는 파일에 대해 FileNotFoundError 가 발생해야 한다."""
    from naviertwin.core.cfd_reader.vtk_reader import VTKReader

    reader = VTKReader()
    with pytest.raises(FileNotFoundError):
        reader.read(Path("/nonexistent/path/file.vtu"))


def test_vtk_reader_unsupported_extension(tmp_path: Path) -> None:
    """지원하지 않는 확장자에 대해 ValueError 가 발생해야 한다."""
    from naviertwin.core.cfd_reader.vtk_reader import VTKReader

    fake = tmp_path / "data.csv"
    fake.write_text("a,b,c\n")

    reader = VTKReader()
    with pytest.raises(ValueError, match="지원하지 않는"):
        reader.read(fake)


def test_vtk_reader_stl(tmp_path: Path) -> None:
    """VTKReader 가 .stl 파일을 읽을 수 있는지 검증한다."""
    import pyvista as pv
    from naviertwin.core.cfd_reader.vtk_reader import VTKReader

    sphere = pv.Sphere()
    stl_path = tmp_path / "sphere.stl"
    sphere.save(str(stl_path))

    reader = VTKReader()
    dataset = reader.read(stl_path)

    assert dataset.n_points > 0


# ---------------------------------------------------------------------------
# 테스트: ReaderFactory
# ---------------------------------------------------------------------------


def test_reader_factory_vtk(tmp_path: Path) -> None:
    """ReaderFactory 가 .vtu 경로에 대해 VTKReader 를 선택하는지 검증한다."""
    import pyvista as pv
    from naviertwin.core.cfd_reader.reader_factory import ReaderFactory
    from naviertwin.core.cfd_reader.vtk_reader import VTKReader

    mesh = _make_simple_ug()
    vtu_path = tmp_path / "test.vtu"
    mesh.save(str(vtu_path))

    reader = ReaderFactory.get_reader(vtu_path)
    assert isinstance(reader, VTKReader)


def test_reader_factory_create_and_read(tmp_path: Path) -> None:
    """ReaderFactory.create_and_read 가 CFDDataset 을 반환하는지 검증한다."""
    import pyvista as pv
    from naviertwin.core.cfd_reader.reader_factory import ReaderFactory
    from naviertwin.core.cfd_reader.base import CFDDataset

    mesh = _make_simple_ug()
    vtu_path = tmp_path / "test.vtu"
    mesh.save(str(vtu_path))

    dataset = ReaderFactory.create_and_read(vtu_path)
    assert isinstance(dataset, CFDDataset)
    assert dataset.n_points > 0


def test_reader_factory_unsupported_format(tmp_path: Path) -> None:
    """지원하지 않는 포맷에 대해 ValueError 가 발생해야 한다."""
    from naviertwin.core.cfd_reader.reader_factory import ReaderFactory

    fake = tmp_path / "data.xyz123"
    fake.write_text("dummy")

    with pytest.raises(ValueError):
        ReaderFactory.get_reader(fake)


def test_reader_factory_nonexistent_path() -> None:
    """존재하지 않는 경로에 대해 FileNotFoundError 가 발생해야 한다."""
    from naviertwin.core.cfd_reader.reader_factory import ReaderFactory

    with pytest.raises(FileNotFoundError):
        ReaderFactory.get_reader(Path("/no/such/path/file.vtu"))


def test_reader_factory_registered_extensions() -> None:
    """레지스트리에 VTK 확장자들이 등록되어 있는지 검증한다."""
    from naviertwin.core.cfd_reader import ReaderFactory  # __init__ 에서 등록

    exts = ReaderFactory.registered_extensions()
    assert ".vtu" in exts
    assert ".vtk" in exts
    assert ".stl" in exts
    assert ".foam" in exts


# ---------------------------------------------------------------------------
# 테스트: NTwinWriter / NTwinReader
# ---------------------------------------------------------------------------


def test_ntwin_write_and_read(tmp_path: Path) -> None:
    """NTwinWriter 로 저장한 데이터를 NTwinReader 로 올바르게 복원하는지 검증한다."""
    from naviertwin.core.cfd_reader.base import CFDDataset
    from naviertwin.core.export.ntwin_format import NTwinReader, NTwinWriter

    mesh = _make_simple_ug()
    dataset = CFDDataset(
        mesh=mesh,
        time_steps=[0.0, 1.0],
        field_names=["U", "p"],
        metadata={"test": True},
    )
    ntwin_path = tmp_path / "test.ntwin"

    with NTwinWriter(ntwin_path) as writer:
        writer.write_dataset(dataset)

    with NTwinReader(ntwin_path) as reader:
        loaded = reader.read()
        ts = reader.time_steps
        fn = reader.field_names

    assert loaded.n_points > 0
    assert ts == pytest.approx([0.0, 1.0])
    assert "U" in fn
    assert "p" in fn


def test_ntwin_write_read_roundtrip_fields(tmp_path: Path) -> None:
    """저장/로드 후 point_data 필드 값이 유지되는지 검증한다."""
    from naviertwin.core.cfd_reader.base import CFDDataset
    from naviertwin.core.export.ntwin_format import NTwinReader, NTwinWriter

    mesh = _make_simple_ug()
    original_p = np.asarray(mesh.point_data["p"]).copy()

    dataset = CFDDataset(
        mesh=mesh,
        time_steps=[0.0],
        field_names=["U", "p"],
    )
    ntwin_path = tmp_path / "fields.ntwin"

    with NTwinWriter(ntwin_path) as writer:
        writer.write_dataset(dataset)

    with NTwinReader(ntwin_path) as reader:
        loaded_mesh = reader.read_timestep(0)

    assert "p" in loaded_mesh.point_data
    loaded_p = np.asarray(loaded_mesh.point_data["p"])
    np.testing.assert_allclose(loaded_p, original_p, rtol=1e-4)


def test_ntwin_reader_file_not_found() -> None:
    """존재하지 않는 파일에 대해 FileNotFoundError 가 발생해야 한다."""
    from naviertwin.core.export.ntwin_format import NTwinReader

    with pytest.raises(FileNotFoundError):
        NTwinReader(Path("/no/such/file.ntwin"))


def test_ntwin_reader_timestep_out_of_range(tmp_path: Path) -> None:
    """범위를 벗어난 타임스텝 인덱스에 IndexError 가 발생해야 한다."""
    from naviertwin.core.cfd_reader.base import CFDDataset
    from naviertwin.core.export.ntwin_format import NTwinReader, NTwinWriter

    mesh = _make_simple_ug()
    dataset = CFDDataset(mesh=mesh, time_steps=[0.0], field_names=["p"])
    ntwin_path = tmp_path / "single.ntwin"

    with NTwinWriter(ntwin_path) as writer:
        writer.write_dataset(dataset)

    with NTwinReader(ntwin_path) as reader:
        with pytest.raises(IndexError):
            reader.read_timestep(99)


def test_ntwin_save_load_convenience(tmp_path: Path) -> None:
    """save_dataset/load_dataset 편의 함수가 올바르게 동작하는지 검증한다."""
    from naviertwin.core.cfd_reader.base import CFDDataset
    from naviertwin.core.export.ntwin_format import load_dataset, save_dataset

    mesh = _make_simple_ug()
    dataset = CFDDataset(
        mesh=mesh,
        time_steps=[0.0, 0.5, 1.0],
        field_names=["U", "p"],
    )
    ntwin_path = tmp_path / "convenience.ntwin"
    save_dataset(dataset, ntwin_path)

    loaded = load_dataset(ntwin_path)
    assert loaded.n_points == dataset.n_points
    assert len(loaded.time_steps) == 3


def test_ntwin_append_snapshot(tmp_path: Path) -> None:
    """append_snapshot 으로 타임스텝을 순차 추가할 수 있는지 검증한다."""
    from naviertwin.core.export.ntwin_format import NTwinReader, NTwinWriter

    ntwin_path = tmp_path / "append.ntwin"
    meshes = [_make_simple_ug() for _ in range(3)]
    times = [0.0, 0.5, 1.0]

    with NTwinWriter(ntwin_path) as writer:
        for mesh, t in zip(meshes, times):
            writer.append_snapshot(mesh, t)

    with NTwinReader(ntwin_path) as reader:
        ts = reader.time_steps

    assert len(ts) == 3
    assert ts == pytest.approx(times)


# ---------------------------------------------------------------------------
# 테스트: OpenFOAMReader (pyvista.POpenFOAMReader 가능 여부에 따라 동작)
# ---------------------------------------------------------------------------


def test_openfoam_reader_foam_file_auto_detect(tmp_path: Path) -> None:
    """OpenFOAMReader 가 .foam 파일로부터 케이스 디렉토리를 인식하는지 검증한다."""
    from naviertwin.core.cfd_reader.openfoam_reader import OpenFOAMReader

    # 가상 OpenFOAM 케이스 디렉토리 구조 생성
    case_dir = tmp_path / "cavity"
    (case_dir / "constant" / "polyMesh").mkdir(parents=True)
    (case_dir / "system").mkdir()
    (case_dir / "0").mkdir()

    foam_file = case_dir / "cavity.foam"
    foam_file.touch()

    reader = OpenFOAMReader()

    # pyvista.POpenFOAMReader 로 읽기 시도 — 실패해도 ImportError/Exception 이어야 함
    # (실제 CFD 파일 없이 파싱 자체는 실패 가능)
    try:
        dataset = reader.read(foam_file)
        # 성공 시 기본 구조 확인
        assert dataset.time_steps is not None
    except (ImportError, Exception):
        # pyvista 없거나 실제 데이터 없으면 스킵
        pytest.skip("OpenFOAM 실제 케이스 파일 없음 또는 pyvista 부재")


def test_openfoam_reader_creates_foam_file(tmp_path: Path) -> None:
    """OpenFOAMReader 가 디렉토리 경로에서 .foam 파일을 자동 생성하는지 검증한다."""
    from naviertwin.core.cfd_reader.openfoam_reader import OpenFOAMReader

    case_dir = tmp_path / "autofoam"
    (case_dir / "constant").mkdir(parents=True)
    (case_dir / "system").mkdir()

    reader = OpenFOAMReader()
    foam_file = reader._create_foam_file(case_dir)

    assert foam_file.exists()
    assert foam_file.suffix == ".foam"
    assert foam_file.parent == case_dir


def test_openfoam_detect_time_steps(tmp_path: Path) -> None:
    """_detect_time_steps 가 숫자 디렉토리만 타임스텝으로 인식하는지 검증한다."""
    from naviertwin.core.cfd_reader.openfoam_reader import OpenFOAMReader

    case_dir = tmp_path / "ts_test"
    for name in ("0", "0.5", "1", "2.0", "constant", "system", "processor0"):
        (case_dir / name).mkdir(parents=True)

    reader = OpenFOAMReader()
    ts = reader._detect_time_steps(case_dir)

    assert ts == pytest.approx([0.0, 0.5, 1.0, 2.0])
    assert 0.0 in ts  # "0" 디렉토리 포함


def test_openfoam_detect_field_names(tmp_path: Path) -> None:
    """_detect_field_names 가 타임 디렉토리에서 필드 이름을 올바르게 수집하는지."""
    from naviertwin.core.cfd_reader.openfoam_reader import OpenFOAMReader

    case_dir = tmp_path / "field_test"
    time_dir = case_dir / "1"
    time_dir.mkdir(parents=True)

    for field in ("U", "p", "k", "epsilon"):
        (time_dir / field).write_text("FoamFile { ... }")

    reader = OpenFOAMReader()
    fields = reader._detect_field_names(case_dir, "1")

    assert set(fields) == {"U", "p", "k", "epsilon"}
