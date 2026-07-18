"""DatasetSignature 테스트 — 메쉬 위상/좌표 해시 기반 케이스 식별자 (§6½).

같은 격자/같은 형상 판정을 좌표 전수 비교 대신 해시로 하는
:mod:`naviertwin.core.data_model.signature` 의 계약을 검증한다.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="시그니처 계산에는 pyvista 가 필요합니다.")

import pyvista as pv  # noqa: E402

from naviertwin.core.cfd_reader.base import CFDDataset  # noqa: E402
from naviertwin.core.data_model import (  # noqa: E402
    DatasetSignature,
    assign_geometry_ids,
    compute_signature,
    same_mesh,
)
from naviertwin.core.preprocessing.group_split import group_train_val_test_split  # noqa: E402
from naviertwin.web.service import dataset_signatures, make_demo_case_set  # noqa: E402


def _small_grid(n: int = 10) -> pv.UnstructuredGrid:
    """n×n 균일 격자를 UnstructuredGrid 로 만든다."""
    grid = pv.ImageData(
        dimensions=(n, n, 1),
        spacing=(1.0 / (n - 1), 1.0 / (n - 1), 1.0),
    )
    return grid.cast_to_unstructured_grid()


def _holed_grid(radius: float, n: int = 21) -> pv.UnstructuredGrid:
    """중앙에 반지름 radius 구멍을 뚫은 격자 — 데모 'shapes' 의 원시 메쉬와 동일 방식.

    주의: 반지름 간격이 격자 간격(1/(n-1))보다 작으면 인접 반지름이 같은
    점 집합을 제거해 같은 격자가 된다 — 테스트에서는 간격을 충분히 벌릴 것.
    """
    ug = _small_grid(n)
    coords = np.asarray(ug.points, dtype=np.float64)
    dist = np.linalg.norm(coords[:, :2] - np.array([0.5, 0.5]), axis=1)
    ug.point_data["dist"] = dist
    holed = ug.threshold(radius, scalars="dist")
    holed.point_data.pop("dist", None)
    return holed


# ──────────────────────────────────────────────────────────────────────
# compute_signature / same_mesh
# ──────────────────────────────────────────────────────────────────────


def test_deep_copy_has_identical_signature() -> None:
    """같은 메쉬의 deep copy → 시그니처 완전 동일, same_mesh True."""
    mesh = _small_grid()
    twin = mesh.copy(deep=True)

    sig_a = compute_signature(mesh)
    sig_b = compute_signature(twin)

    assert isinstance(sig_a, DatasetSignature)
    assert sig_a == sig_b  # frozen dataclass — 필드 전부 일치
    assert len(sig_a.topology_hash) == 16
    assert len(sig_a.coordinate_hash) == 16
    assert sig_a.n_points == mesh.n_points
    assert sig_a.n_cells == mesh.n_cells
    assert same_mesh(mesh, twin)
    # 시그니처끼리도 비교 가능해야 한다.
    assert same_mesh(sig_a, sig_b)


def test_accepts_cfddataset_and_image_data() -> None:
    """CFDDataset(.mesh) 과 비-UnstructuredGrid(ImageData) 둘 다 허용한다."""
    image = pv.ImageData(dimensions=(10, 10, 1), spacing=(1.0 / 9, 1.0 / 9, 1.0))
    dataset = CFDDataset(mesh=image, time_steps=[0.0], field_names=[])

    sig_from_dataset = compute_signature(dataset)
    sig_from_cast = compute_signature(image.cast_to_unstructured_grid())

    assert sig_from_dataset == sig_from_cast
    assert same_mesh(dataset, image)


def test_coordinate_shift_changes_only_coordinate_hash() -> None:
    """좌표만 살짝 이동 → coordinate_hash 다름, topology_hash 같음."""
    mesh = _small_grid()
    shifted = mesh.copy(deep=True)
    shifted.points = np.asarray(shifted.points) + 0.001

    sig_ref = compute_signature(mesh)
    sig_shift = compute_signature(shifted)

    assert sig_shift.topology_hash == sig_ref.topology_hash  # 연결성 불변
    assert sig_shift.coordinate_hash != sig_ref.coordinate_hash
    assert not same_mesh(mesh, shifted)


def test_cell_removal_changes_topology_hash() -> None:
    """셀 하나 제거(extract_cells) → topology_hash 다름."""
    mesh = _small_grid()
    reduced = mesh.extract_cells(np.arange(mesh.n_cells - 1))

    sig_ref = compute_signature(mesh)
    sig_cut = compute_signature(reduced)

    assert sig_cut.topology_hash != sig_ref.topology_hash
    assert not same_mesh(mesh, reduced)


# ──────────────────────────────────────────────────────────────────────
# assign_geometry_ids
# ──────────────────────────────────────────────────────────────────────


def test_distinct_shapes_get_distinct_ids() -> None:
    """형상(구멍 반지름)이 전부 다른 5 케이스 → 등장 순서대로 [0,1,2,3,4]."""
    radii = [0.08, 0.16, 0.24, 0.32, 0.40]  # 간격 0.08 > 격자 간격 0.05
    datasets = [
        CFDDataset(mesh=_holed_grid(r), time_steps=[0.0], field_names=[]) for r in radii
    ]
    assert assign_geometry_ids(datasets) == [0, 1, 2, 3, 4]


def test_same_dataset_twice_gets_same_id() -> None:
    """같은 데이터셋을 2번 넣으면 같은 id 를 받는다."""
    a = CFDDataset(mesh=_holed_grid(0.1), time_steps=[0.0], field_names=[])
    b = CFDDataset(mesh=_holed_grid(0.2), time_steps=[0.0], field_names=[])
    assert assign_geometry_ids([a, b, a]) == [0, 1, 0]
    # 별도 객체라도 격자가 같으면 (deep copy) 같은 id.
    a_copy = CFDDataset(mesh=a.mesh.copy(deep=True), time_steps=[0.0], field_names=[])
    assert assign_geometry_ids([a, a_copy]) == [0, 0]


def test_shapes_demo_resampled_collapses_to_single_id() -> None:
    """데모 'shapes' 는 로드 시 공통 격자로 재샘플된다 — 재샘플 **후** 케이스는
    전부 같은 격자이므로 geometry_id 가 하나로 접힌다 (형상 정보는 좌표가
    아니라 sdf field 로 옮겨간 상태다). 원시(재샘플 전) 형상별 id 분리는
    :func:`test_distinct_shapes_get_distinct_ids` 가 같은 구성 방식으로 검증한다."""
    case_set = make_demo_case_set("shapes", n_side=24, resolution=16)
    datasets = case_set["datasets"]
    assert len(datasets) == 5
    assert case_set["resampled"] is True
    assert assign_geometry_ids(datasets) == [0, 0, 0, 0, 0]


# ──────────────────────────────────────────────────────────────────────
# service 래퍼 + 그룹 스플릿 통합 스모크
# ──────────────────────────────────────────────────────────────────────


def test_dataset_signatures_wrapper_returns_dicts() -> None:
    """web.service.dataset_signatures — dataclass 를 UI/저장용 dict 로 푼다."""
    datasets = [
        CFDDataset(mesh=_holed_grid(0.1), time_steps=[0.0], field_names=[]),
        CFDDataset(mesh=_holed_grid(0.25), time_steps=[0.0], field_names=[]),
    ]
    rows = dataset_signatures(datasets)
    assert len(rows) == 2
    for row, ds in zip(rows, datasets):
        assert set(row) == {"topology_hash", "coordinate_hash", "n_points", "n_cells"}
        assert row["n_points"] == ds.n_points
        assert row["n_cells"] == ds.n_cells
    assert rows[0]["topology_hash"] != rows[1]["topology_hash"]


def test_geometry_ids_feed_group_split() -> None:
    """assign_geometry_ids 결과를 group_train_val_test_split 에 바로 넣는 스모크.

    형상 4종 × 각 2케이스(운전조건만 다름 = 같은 격자) 총 8케이스 —
    같은 geometry_id 그룹은 절대 train/val/test 를 가로지르지 않아야 한다.
    """
    radii = [0.10, 0.20, 0.30, 0.40]  # 간격 0.10 > 격자 간격 0.05
    datasets = []
    for r in radii:
        base = _holed_grid(r)
        for _ in range(2):  # 같은 형상의 케이스 2개 (deep copy — 같은 격자)
            datasets.append(
                CFDDataset(mesh=base.copy(deep=True), time_steps=[0.0], field_names=[])
            )

    ids = assign_geometry_ids(datasets)
    assert ids == [0, 0, 1, 1, 2, 2, 3, 3]

    split = group_train_val_test_split(
        n_cases=len(datasets), group_ids=ids, val_frac=0.25, test_frac=0.25, seed=0
    )
    all_idx = np.concatenate([split.train_idx, split.val_idx, split.test_idx])
    assert sorted(all_idx.tolist()) == list(range(len(datasets)))

    # 같은 그룹이 두 분할에 걸치지 않는다.
    id_arr = np.asarray(ids)
    groups_per_split = [
        set(id_arr[split.train_idx].tolist()),
        set(id_arr[split.val_idx].tolist()),
        set(id_arr[split.test_idx].tolist()),
    ]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not (groups_per_split[i] & groups_per_split[j])
