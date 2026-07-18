"""메쉬 기반 벽면 특징(mesh_features) 및 OpenFOAM 패치 보존 테스트.

실제 CFD 파일 없이 합성 메쉬(pv.ImageData 볼륨 + 닫힌 구/열린 평면
PolyData 벽면)만으로 동작한다. pyvista 가 없으면 전체가 skip 된다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista", reason="pyvista 가 설치되어 있지 않음")


# ---------------------------------------------------------------------------
# 헬퍼: 합성 볼륨 메쉬 + 벽면
# ---------------------------------------------------------------------------

_SPACING = 0.2


def _make_volume_grid() -> "pv.ImageData":
    """[-1.4, 1.4]^3 을 덮는 15^3 ImageData 볼륨을 생성한다."""
    return pv.ImageData(
        dimensions=(15, 15, 15),
        spacing=(_SPACING, _SPACING, _SPACING),
        origin=(-1.4, -1.4, -1.4),
    )


def _make_closed_sphere(radius: float = 0.5) -> "pv.PolyData":
    """원점 중심의 닫힌 구 벽면을 생성한다."""
    return pv.Sphere(radius=radius, theta_resolution=48, phi_resolution=48)


def _make_dataset(mesh: "pv.DataSet") -> "object":
    """pyvista 메쉬로 CFDDataset 을 생성한다."""
    from naviertwin.core.cfd_reader.base import CFDDataset

    ug = mesh.cast_to_unstructured_grid()
    n = ug.n_points
    ug.point_data["p"] = np.random.rand(n).astype(np.float32)
    return CFDDataset(
        mesh=ug,
        time_steps=[0.0],
        field_names=["p"],
        metadata={"reader": "synthetic"},
    )


# ---------------------------------------------------------------------------
# 테스트: wall_distance
# ---------------------------------------------------------------------------


def test_wall_distance_nonnegative_and_finite() -> None:
    """wall_distance 는 모든 점에서 비음수·유한이어야 한다."""
    from naviertwin.core.geometry.mesh_features import wall_distance

    grid = _make_volume_grid()
    sphere = _make_closed_sphere()

    d = wall_distance(grid, sphere)

    assert d.shape == (grid.n_points,)
    assert np.all(np.isfinite(d))
    assert np.all(d >= 0.0)


def test_wall_distance_zero_near_wall_and_grows_away() -> None:
    """벽 근처 점에서는 거의 0, 멀어질수록 커져야 한다."""
    from naviertwin.core.geometry.mesh_features import wall_distance

    grid = _make_volume_grid()
    radius = 0.5
    sphere = _make_closed_sphere(radius=radius)

    d = wall_distance(grid, sphere)
    r = np.linalg.norm(grid.points, axis=1)

    # 벽면(r ≈ radius)에서 한 셀 이내를 지나는 점이 반드시 존재한다.
    near_wall = np.abs(r - radius) <= _SPACING / 2
    assert near_wall.any()
    assert d[near_wall].max() <= _SPACING

    # 해석적 거리 | r - radius | 와 일치해야 한다 (구 이산화 오차 허용).
    analytic = np.abs(r - radius)
    np.testing.assert_allclose(d, analytic, atol=0.02)

    # 멀리 있는 점의 거리가 벽 근처 점보다 커야 한다.
    far = r > 1.2
    assert d[far].min() > d[near_wall].max()


# ---------------------------------------------------------------------------
# 테스트: signed_distance (닫힌 구 — 내부 음수 / 외부 양수)
# ---------------------------------------------------------------------------


def test_signed_distance_sign_convention_closed_sphere() -> None:
    """닫힌 구에서 내부는 음수, 외부는 양수여야 한다."""
    from naviertwin.core.geometry.mesh_features import (
        signed_distance,
        wall_distance,
    )

    grid = _make_volume_grid()
    radius = 0.5
    sphere = _make_closed_sphere(radius=radius)

    s = signed_distance(grid, sphere)
    r = np.linalg.norm(grid.points, axis=1)

    inside = r < radius - _SPACING / 2
    outside = r > radius + _SPACING / 2
    assert inside.any() and outside.any()
    assert np.all(s[inside] < 0.0)
    assert np.all(s[outside] > 0.0)

    # 크기는 wall_distance 와 일치해야 한다.
    np.testing.assert_allclose(np.abs(s), wall_distance(grid, sphere))


# ---------------------------------------------------------------------------
# 테스트: surface_features
# ---------------------------------------------------------------------------


def test_surface_features_shapes_and_area() -> None:
    """법선/곡률/면적 배열의 shape 와 값 범위를 검증한다."""
    from naviertwin.core.geometry.mesh_features import surface_features

    radius = 0.5
    sphere = _make_closed_sphere(radius=radius)

    feats = surface_features(sphere)

    normals = feats["normals"]
    curvature = feats["curvature"]
    area = feats["area"]

    assert normals.shape == (sphere.n_points, 3)
    np.testing.assert_allclose(
        np.linalg.norm(normals, axis=1), 1.0, atol=1e-5
    )

    assert curvature.shape == (sphere.n_points,)
    assert np.all(np.isfinite(curvature))
    # 구의 평균 곡률 ≈ 1/r (이산화 오차 허용).
    assert np.median(curvature) == pytest.approx(1.0 / radius, rel=0.15)

    assert area.shape == (sphere.n_cells,)
    assert np.all(area > 0.0)
    # 총 면적 ≈ 4 π r² (이산화로 약간 작다).
    assert area.sum() == pytest.approx(4.0 * np.pi * radius**2, rel=0.05)


# ---------------------------------------------------------------------------
# 테스트: attach_wall_features
# ---------------------------------------------------------------------------


def test_attach_wall_features_closed_surface_attaches_sdf() -> None:
    """닫힌 벽면이면 distance 와 sdf 두 필드를 모두 부착해야 한다."""
    from naviertwin.core.geometry.mesh_features import attach_wall_features

    dataset = _make_dataset(_make_volume_grid())
    sphere = _make_closed_sphere()

    names = attach_wall_features(dataset, sphere)

    assert names == ["wall_distance", "wall_sdf"]
    for name in names:
        assert name in dataset.mesh.point_data
        assert name in dataset.field_names
        values = np.asarray(dataset.mesh.point_data[name])
        assert values.shape == (dataset.n_points,)
        assert np.all(np.isfinite(values))

    assert np.all(np.asarray(dataset.mesh.point_data["wall_distance"]) >= 0.0)
    # 구 내부 점이 존재하므로 SDF 에는 음수 값이 있어야 한다.
    assert np.asarray(dataset.mesh.point_data["wall_sdf"]).min() < 0.0


def test_attach_wall_features_open_surface_skips_sdf() -> None:
    """열린 벽면(평면)이면 sdf 를 생략하고 distance 만 부착해야 한다."""
    from naviertwin.core.geometry.mesh_features import attach_wall_features

    dataset = _make_dataset(_make_volume_grid())
    plane = pv.Plane(center=(0.0, 0.0, 0.0), i_size=2.0, j_size=2.0)

    names = attach_wall_features(dataset, plane, prefix="floor")

    assert names == ["floor_distance"]
    assert "floor_distance" in dataset.mesh.point_data
    assert "floor_sdf" not in dataset.mesh.point_data
    assert "floor_distance" in dataset.field_names


def test_attach_wall_features_no_duplicate_field_names() -> None:
    """같은 prefix 로 두 번 부착해도 field_names 가 중복되지 않아야 한다."""
    from naviertwin.core.geometry.mesh_features import attach_wall_features

    dataset = _make_dataset(_make_volume_grid())
    sphere = _make_closed_sphere()

    attach_wall_features(dataset, sphere)
    attach_wall_features(dataset, sphere)

    assert dataset.field_names.count("wall_distance") == 1
    assert dataset.field_names.count("wall_sdf") == 1


# ---------------------------------------------------------------------------
# 테스트: wall_surface_from_patches
# ---------------------------------------------------------------------------


def _dataset_with_patches() -> tuple["object", "pv.PolyData", "pv.PolyData"]:
    """boundary_patch_meshes 메타데이터를 가진 데이터셋을 만든다."""
    dataset = _make_dataset(_make_volume_grid())
    lower = pv.Plane(center=(0.0, 0.0, -1.0), i_size=2.0, j_size=2.0)
    upper = pv.Plane(center=(0.0, 0.0, 1.0), i_size=2.0, j_size=2.0)
    dataset.metadata["boundary_patch_meshes"] = {
        "lowerWall": lower,
        "upperWall": upper,
    }
    dataset.metadata["boundary_patches"] = {
        "lowerWall": {"n_cells": lower.n_cells, "n_points": lower.n_points},
        "upperWall": {"n_cells": upper.n_cells, "n_points": upper.n_points},
    }
    return dataset, lower, upper


def test_wall_surface_from_patches_merges_selected() -> None:
    """선택한 두 패치가 하나의 PolyData 로 병합되어야 한다."""
    from naviertwin.core.geometry.mesh_features import wall_surface_from_patches

    dataset, lower, upper = _dataset_with_patches()

    merged = wall_surface_from_patches(dataset, ["lowerWall", "upperWall"])

    assert isinstance(merged, pv.PolyData)
    assert merged.n_cells == lower.n_cells + upper.n_cells


def test_wall_surface_from_patches_single_patch() -> None:
    """패치 1개 선택 시 해당 표면만 반환해야 한다."""
    from naviertwin.core.geometry.mesh_features import wall_surface_from_patches

    dataset, lower, _ = _dataset_with_patches()

    merged = wall_surface_from_patches(dataset, ["lowerWall"])

    assert isinstance(merged, pv.PolyData)
    assert merged.n_cells == lower.n_cells


def test_wall_surface_from_patches_unknown_name_raises() -> None:
    """알 수 없는 패치 이름이면 ValueError 가 발생해야 한다."""
    from naviertwin.core.geometry.mesh_features import wall_surface_from_patches

    dataset, _, _ = _dataset_with_patches()

    with pytest.raises(ValueError, match="알 수 없는 패치 이름"):
        wall_surface_from_patches(dataset, ["lowerWall", "nope"])


def test_wall_surface_from_patches_missing_metadata_raises() -> None:
    """boundary_patch_meshes 메타데이터가 없으면 ValueError 가 발생해야 한다."""
    from naviertwin.core.geometry.mesh_features import wall_surface_from_patches

    dataset = _make_dataset(_make_volume_grid())

    with pytest.raises(ValueError, match="boundary_patch_meshes"):
        wall_surface_from_patches(dataset, ["anything"])


def test_wall_surface_from_patches_empty_names_raises() -> None:
    """patch_names 가 비어 있으면 ValueError 가 발생해야 한다."""
    from naviertwin.core.geometry.mesh_features import wall_surface_from_patches

    dataset, _, _ = _dataset_with_patches()

    with pytest.raises(ValueError, match="patch_names 가 비어"):
        wall_surface_from_patches(dataset, [])


def test_wall_surface_from_patches_output_usable_for_features() -> None:
    """병합 결과가 attach_wall_features 입력으로 바로 사용 가능해야 한다."""
    from naviertwin.core.geometry.mesh_features import (
        attach_wall_features,
        wall_surface_from_patches,
    )

    dataset, _, _ = _dataset_with_patches()
    wall = wall_surface_from_patches(dataset, ["lowerWall", "upperWall"])

    names = attach_wall_features(dataset, wall)

    # 두 평면을 병합한 표면은 열린 표면이므로 distance 만 부착된다.
    assert names == ["wall_distance"]
    assert np.all(
        np.isfinite(np.asarray(dataset.mesh.point_data["wall_distance"]))
    )


# ---------------------------------------------------------------------------
# 테스트: OpenFOAM 리더 패치 보존 헬퍼 (합성 MultiBlock)
# ---------------------------------------------------------------------------


def test_extract_internal_mesh_prefers_internal_block() -> None:
    """internalMesh 블록이 있으면 그것만 추출해야 한다 (경계 병합 금지)."""
    from naviertwin.core.cfd_reader.openfoam_reader import (
        _extract_internal_mesh,
    )

    internal = _make_volume_grid().cast_to_unstructured_grid()
    boundary = pv.MultiBlock()
    boundary["wall"] = pv.Plane()

    mb = pv.MultiBlock()
    mb["internalMesh"] = internal
    mb["boundary"] = boundary

    mesh = _extract_internal_mesh(mb)

    assert isinstance(mesh, pv.UnstructuredGrid)
    assert mesh.n_points == internal.n_points
    assert mesh.n_cells == internal.n_cells


def test_extract_internal_mesh_falls_back_to_combine() -> None:
    """internalMesh 블록이 없으면 기존 combine 동작으로 폴백해야 한다."""
    from naviertwin.core.cfd_reader.openfoam_reader import (
        _extract_internal_mesh,
    )

    ug = _make_volume_grid().cast_to_unstructured_grid()
    mb = pv.MultiBlock()
    mb["something"] = ug

    mesh = _extract_internal_mesh(mb)

    assert isinstance(mesh, pv.UnstructuredGrid)
    assert mesh.n_points == ug.n_points


def test_extract_boundary_patches_from_multiblock() -> None:
    """boundary 중첩 MultiBlock 에서 패치별 PolyData 를 추출해야 한다."""
    from naviertwin.core.cfd_reader.openfoam_reader import (
        _extract_boundary_patches,
    )

    inlet = pv.Plane(center=(-1.0, 0.0, 0.0))
    outlet = pv.Plane(center=(1.0, 0.0, 0.0))
    boundary = pv.MultiBlock()
    boundary["inlet"] = inlet
    boundary["outlet"] = outlet

    mb = pv.MultiBlock()
    mb["internalMesh"] = _make_volume_grid().cast_to_unstructured_grid()
    mb["boundary"] = boundary

    patches = _extract_boundary_patches(mb)

    assert set(patches) == {"inlet", "outlet"}
    for patch in patches.values():
        assert isinstance(patch, pv.PolyData)
        assert patch.n_cells > 0
    assert patches["inlet"].n_cells == inlet.n_cells


def test_extract_boundary_patches_absent_returns_empty() -> None:
    """boundary 블록이 없으면 빈 dict 를 반환해야 한다 (기존 동작 보존)."""
    from naviertwin.core.cfd_reader.openfoam_reader import (
        _extract_boundary_patches,
    )

    mb = pv.MultiBlock()
    mb["internalMesh"] = _make_volume_grid().cast_to_unstructured_grid()

    assert _extract_boundary_patches(mb) == {}
    assert _extract_boundary_patches(None) == {}
    assert _extract_boundary_patches(pv.Sphere()) == {}


# ---------------------------------------------------------------------------
# 테스트: 실제 OpenFOAM 케이스 패치 보존 (fixture 존재 시에만)
# ---------------------------------------------------------------------------


def test_openfoam_reader_preserves_patches_on_real_case() -> None:
    """실제 OpenFOAM 케이스에서 boundary_patches 메타데이터가 보존되는지."""
    fixture = Path(__file__).parent / "fixtures" / "openfoam_case"
    if not (fixture / "constant" / "polyMesh" / "boundary").exists():
        pytest.skip(
            "저장소에 OpenFOAM polyMesh 샘플 케이스가 없음 "
            "(tests/fixtures/openfoam_case). 실제 케이스 패치 보존 테스트 생략."
        )

    from naviertwin.core.cfd_reader.openfoam_reader import OpenFOAMReader

    dataset = OpenFOAMReader().read(fixture)

    assert "boundary_patches" in dataset.metadata
    assert "boundary_patch_meshes" in dataset.metadata
    patches = dataset.metadata["boundary_patches"]
    meshes = dataset.metadata["boundary_patch_meshes"]
    assert set(patches) == set(meshes)
    for name, info in patches.items():
        assert info["n_cells"] == meshes[name].n_cells
        assert info["n_points"] == meshes[name].n_points
