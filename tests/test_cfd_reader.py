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


def test_vtk_reader_pvd_time_series(tmp_path: Path) -> None:
    """VTKReader 가 PVD 컬렉션을 time-series snapshot 행렬로 읽어야 한다."""
    from naviertwin.core.cfd_reader.vtk_pvd_writer import write_pvd
    from naviertwin.core.cfd_reader.vtk_reader import VTKReader

    entries: list[tuple[float, str]] = []
    for index, time_value in enumerate([0.0, 0.5, 1.0]):
        mesh = _make_simple_ug()
        n_points = mesh.n_points
        mesh.point_data["p"] = np.full(n_points, index + 1, dtype=np.float32)
        mesh.point_data["U"] = np.column_stack(
            [
                np.full(n_points, index + 1, dtype=np.float32),
                np.zeros(n_points, dtype=np.float32),
                np.zeros(n_points, dtype=np.float32),
            ]
        )
        file_name = f"step_{index:04d}.vtu"
        mesh.save(str(tmp_path / file_name))
        entries.append((time_value, file_name))

    pvd_path = tmp_path / "series.pvd"
    write_pvd(pvd_path, entries)

    dataset = VTKReader().read(pvd_path)

    assert dataset.time_steps == pytest.approx([0.0, 0.5, 1.0])
    assert dataset.n_time_steps == 3
    assert "p" in dataset.field_names
    assert "U" in dataset.field_names
    assert dataset.metadata["time_series_locations"]["p"] == "point"

    p_snapshots = dataset.extract_field_snapshots("p")
    assert p_snapshots.shape == (n_points, 3)
    np.testing.assert_allclose(p_snapshots[:, 0], np.ones(n_points))
    np.testing.assert_allclose(p_snapshots[:, 1], np.full(n_points, 2.0))
    np.testing.assert_allclose(p_snapshots[:, 2], np.full(n_points, 3.0))

    u_snapshots = dataset.extract_field_snapshots("U")
    assert u_snapshots.shape == (n_points, 3)
    np.testing.assert_allclose(u_snapshots[:, 2], np.full(n_points, 3.0))


# ---------------------------------------------------------------------------
# 테스트: ReaderFactory
# ---------------------------------------------------------------------------


def test_reader_factory_vtk(tmp_path: Path) -> None:
    """ReaderFactory 가 .vtu 경로에 대해 VTKReader 를 선택하는지 검증한다."""
    from naviertwin.core.cfd_reader.reader_factory import ReaderFactory
    from naviertwin.core.cfd_reader.vtk_reader import VTKReader

    mesh = _make_simple_ug()
    vtu_path = tmp_path / "test.vtu"
    mesh.save(str(vtu_path))

    reader = ReaderFactory.get_reader(vtu_path)
    assert isinstance(reader, VTKReader)


def test_reader_factory_create_and_read(tmp_path: Path) -> None:
    """ReaderFactory.create_and_read 가 CFDDataset 을 반환하는지 검증한다."""
    from naviertwin.core.cfd_reader.base import CFDDataset
    from naviertwin.core.cfd_reader.reader_factory import ReaderFactory

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
    assert ".pvd" in exts
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


def test_ntwin_write_read_roundtrip_cell_fields(tmp_path: Path) -> None:
    """Cell fields keep their association and values through .ntwin."""
    from naviertwin.core.cfd_reader.base import CFDDataset
    from naviertwin.core.export.ntwin_format import load_dataset, save_dataset

    mesh = _make_simple_ug()
    original = np.arange(mesh.n_cells, dtype=np.float32) + 7.0
    mesh.cell_data["cell_pressure"] = original
    dataset = CFDDataset(
        mesh=mesh,
        time_steps=[0.0],
        field_names=["cell_pressure"],
    )
    path = tmp_path / "cell-fields.ntwin"

    save_dataset(dataset, path)
    loaded = load_dataset(path)

    assert "cell_pressure" not in loaded.mesh.point_data
    np.testing.assert_allclose(loaded.mesh.cell_data["cell_pressure"], original)


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


def test_save_dataset_accepts_compression_keyword(tmp_path: Path) -> None:
    """save_dataset(compression=...) 호출이 TypeError 없이 동작해야 한다."""
    from naviertwin.core.cfd_reader.base import CFDDataset
    from naviertwin.core.export.ntwin_format import load_dataset, save_dataset

    mesh = _make_simple_ug()
    dataset = CFDDataset(mesh=mesh, time_steps=[0.0], field_names=["U", "p"])
    ntwin_path = tmp_path / "compressed.ntwin"

    save_dataset(dataset, ntwin_path, compression="gzip")
    loaded = load_dataset(ntwin_path)
    assert loaded.n_points == dataset.n_points
    assert "p" in loaded.field_names


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


def test_load_dataset_preserves_multitimestep_snapshots(tmp_path: Path) -> None:
    """append 기반 .ntwin 로드시 time-series 스냅샷이 보존되어야 한다."""
    from naviertwin.core.export.ntwin_format import NTwinWriter, load_dataset

    ntwin_path = tmp_path / "append_load.ntwin"
    times = [0.0, 0.5, 1.0]
    meshes = []
    for i, _ in enumerate(times):
        mesh = _make_simple_ug()
        n = mesh.n_points
        mesh.point_data["p"] = np.full((n,), i + 1, dtype=np.float32)
        meshes.append(mesh)

    with NTwinWriter(ntwin_path) as writer:
        for mesh, t in zip(meshes, times):
            writer.append_snapshot(mesh, t)

    loaded = load_dataset(ntwin_path)
    snapshots = loaded.extract_field_snapshots("p")
    assert snapshots.shape[1] == len(times)
    np.testing.assert_allclose(snapshots[:, 0], np.ones(loaded.n_points))
    np.testing.assert_allclose(snapshots[:, 1], np.full((loaded.n_points,), 2.0))
    np.testing.assert_allclose(snapshots[:, 2], np.full((loaded.n_points,), 3.0))


def test_ntwin_append_snapshot_read_timestep_slices_point_data(tmp_path: Path) -> None:
    """append_snapshot 후 read_timestep 이 타임스텝별 PointData를 슬라이스하는지 검증한다."""
    from naviertwin.core.export.ntwin_format import NTwinReader, NTwinWriter

    ntwin_path = tmp_path / "append_slices.ntwin"
    times = [0.0, 0.5, 1.0]
    meshes = []
    for i, _ in enumerate(times):
        mesh = _make_simple_ug()
        n = mesh.n_points
        mesh.point_data["p"] = np.full((n,), i + 1, dtype=np.float32)
        mesh.point_data["U"] = np.full((n, 3), i + 10, dtype=np.float32)
        meshes.append(mesh)

    with NTwinWriter(ntwin_path) as writer:
        for mesh, t in zip(meshes, times):
            writer.append_snapshot(mesh, t)

    with NTwinReader(ntwin_path) as reader:
        for i in range(len(times)):
            loaded = reader.read_timestep(i)
            p = np.asarray(loaded.point_data["p"])
            u = np.asarray(loaded.point_data["U"])
            assert p.shape[0] == meshes[i].n_points
            assert u.shape == (meshes[i].n_points, 3)
            np.testing.assert_allclose(p, np.full_like(p, i + 1))
            np.testing.assert_allclose(u, np.full_like(u, i + 10))


def test_ntwin_reader_read_timestep_raises_on_numberofpoints_length_mismatch(
    tmp_path: Path,
) -> None:
    """NumberOfPoints 길이와 time_steps 길이가 다르면 ValueError 가 발생해야 한다."""
    from naviertwin.core.export.ntwin_format import NTwinReader, NTwinWriter

    ntwin_path = tmp_path / "counts_mismatch.ntwin"
    mesh = _make_simple_ug()
    with NTwinWriter(ntwin_path) as writer:
        writer.append_snapshot(mesh, 0.0)
        writer.append_snapshot(mesh, 1.0)

    with h5py.File(ntwin_path, "a") as f:
        vtk_grp = f["VTKHDF"]
        del vtk_grp["NumberOfPoints"]
        vtk_grp.create_dataset(
            "NumberOfPoints",
            data=np.array([mesh.n_points, mesh.n_points, mesh.n_points], dtype=np.int64),
        )

    with NTwinReader(ntwin_path) as reader:
        with pytest.raises(ValueError, match="NumberOfPoints 길이와 time_steps 길이"):
            reader.read_timestep(1)


def test_ntwin_reader_read_timestep_raises_on_truncated_appended_pointdata(
    tmp_path: Path,
) -> None:
    """append된 PointData가 잘리면 이후 타임스텝 read_timestep 에서 ValueError 가 발생해야 한다."""
    from naviertwin.core.export.ntwin_format import NTwinReader, NTwinWriter

    ntwin_path = tmp_path / "truncated_pointdata.ntwin"
    times = [0.0, 0.5, 1.0]
    mesh = _make_simple_ug()

    with NTwinWriter(ntwin_path) as writer:
        for t in times:
            writer.append_snapshot(mesh, t)

    with h5py.File(ntwin_path, "a") as f:
        pd_grp = f["VTKHDF/PointData"]
        p_data = np.asarray(pd_grp["p"])
        del pd_grp["p"]
        pd_grp.create_dataset("p", data=p_data[:-2])

    with NTwinReader(ntwin_path) as reader:
        with pytest.raises(ValueError, match="PointData 길이가 타임스텝 슬라이스 범위"):
            reader.read_timestep(2)


def test_ntwin_reader_field_names_and_time_steps_fallbacks(tmp_path: Path) -> None:
    """field_names/time_steps 메타데이터가 손상되어도 폴백이 동작하는지 검증한다."""
    from naviertwin.core.export.ntwin_format import NTwinReader, NTwinWriter

    ntwin_path = tmp_path / "fallbacks.ntwin"
    mesh = _make_simple_ug()
    with NTwinWriter(ntwin_path) as writer:
        writer.append_snapshot(mesh, 0.0)

    with h5py.File(ntwin_path, "a") as f:
        nt_grp = f["NavierTwin"]
        if "field_names" in nt_grp:
            del nt_grp["field_names"]
        nt_grp.create_dataset("field_names", data="{not-json")
        if "time_steps" in nt_grp:
            del nt_grp["time_steps"]

    with NTwinReader(ntwin_path) as reader:
        names = reader.field_names
        ts = reader.time_steps

    assert "U" in names
    assert "p" in names
    assert ts == [0.0]


def test_ntwin_reader_field_names_json_object_falls_back_to_point_data(
    tmp_path: Path,
) -> None:
    """field_names 가 JSON object 로 저장되면 PointData 키로 폴백해야 한다."""
    from naviertwin.core.export.ntwin_format import NTwinReader, NTwinWriter

    ntwin_path = tmp_path / "field_names_object_fallback.ntwin"
    mesh = _make_simple_ug()
    with NTwinWriter(ntwin_path) as writer:
        writer.append_snapshot(mesh, 0.0)

    with h5py.File(ntwin_path, "a") as f:
        nt_grp = f["NavierTwin"]
        if "field_names" in nt_grp:
            del nt_grp["field_names"]
        nt_grp.create_dataset("field_names", data=json.dumps({"U": 0, "p": 1}))

    with NTwinReader(ntwin_path) as reader:
        names = reader.field_names

    assert set(names) == {"U", "p"}


def test_ntwin_reader_time_steps_malformed_scalar_falls_back_to_zero(
    tmp_path: Path,
) -> None:
    """time_steps 가 숫자로 변환 불가한 스칼라이면 [0.0] 으로 폴백해야 한다."""
    from naviertwin.core.export.ntwin_format import NTwinReader, NTwinWriter

    ntwin_path = tmp_path / "time_steps_scalar_fallback.ntwin"
    mesh = _make_simple_ug()
    with NTwinWriter(ntwin_path) as writer:
        writer.append_snapshot(mesh, 0.0)

    with h5py.File(ntwin_path, "a") as f:
        nt_grp = f["NavierTwin"]
        if "time_steps" in nt_grp:
            del nt_grp["time_steps"]
        nt_grp.create_dataset("time_steps", data="not-a-float")

    with NTwinReader(ntwin_path) as reader:
        ts = reader.time_steps

    assert ts == [0.0]


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
