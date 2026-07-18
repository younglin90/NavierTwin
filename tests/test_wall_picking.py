"""v5.1 wall face 픽킹(경계 패치 메타 없는 raw 메쉬용 폴백) 테스트.

실제 GL 피킹(라이브 렌더 루프)은 헤드리스 환경에서 재현할 수 없으므로,
서비스 계층(:mod:`naviertwin.web.service`)의 순수 함수와, app 콜백에
피킹 결과를 직접 주입하는 "정직한" 헤드리스 경로만 검증한다
(``st.nt_bc_picked_cells`` 를 실제 피킹 없이 직접 세팅 — test_mesh_features.py
의 합성 메쉬 스타일을 따른다).
"""

from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista", reason="pyvista 가 설치되어 있지 않음")
pytest.importorskip("trame", reason="app 레벨 테스트에는 trame 이 필요합니다.")


# ---------------------------------------------------------------------------
# 헬퍼: 합성 볼륨 메쉬(정육면체 격자) + 표면 셀 선택
# ---------------------------------------------------------------------------


def _make_grid(n: int = 6) -> "pv.ImageData":
    """[0, n-1]^3 정육면체를 덮는 n^3 ImageData 볼륨을 생성한다."""
    return pv.ImageData(dimensions=(n, n, n), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))


def _make_dataset(mesh: "pv.DataSet"):
    """pyvista 메쉬로 CFDDataset 을 생성한다 (test_mesh_features.py 와 동일 스타일)."""
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


def _face_cell_ids(surface: "pv.PolyData", axis: int, value: float, tol: float = 1e-6) -> list[int]:
    """표면 셀 중 지정한 축=value 평면에 놓인 셀 id 목록을 찾는다."""
    centers = surface.cell_centers().points
    mask = np.abs(np.asarray(centers)[:, axis] - value) <= tol
    return np.nonzero(mask)[0].tolist()


# ---------------------------------------------------------------------------
# service: wall_surface_from_picked_cells
# ---------------------------------------------------------------------------


def test_wall_surface_from_picked_cells_returns_face_polydata() -> None:
    from naviertwin.web.service import wall_surface_from_picked_cells

    dataset = _make_dataset(_make_grid())
    surface = dataset.mesh.extract_surface()
    ids = _face_cell_ids(surface, axis=0, value=0.0)
    assert ids, "정육면체 x=0 면에 표면 셀이 있어야 한다"

    wall = wall_surface_from_picked_cells(dataset, ids)

    assert isinstance(wall, pv.PolyData)
    # 표면 추출 과정에서 삼각분할 등으로 셀 수가 달라질 수 있으므로 범위만 확인.
    assert wall.n_cells > 0
    assert wall.n_cells <= len(ids)


def test_wall_surface_from_picked_cells_empty_raises() -> None:
    from naviertwin.web.service import wall_surface_from_picked_cells

    dataset = _make_dataset(_make_grid())

    with pytest.raises(ValueError):
        wall_surface_from_picked_cells(dataset, [])


def test_wall_surface_from_picked_cells_out_of_range_raises() -> None:
    from naviertwin.web.service import wall_surface_from_picked_cells

    dataset = _make_dataset(_make_grid())

    with pytest.raises(ValueError):
        wall_surface_from_picked_cells(dataset, [10**9])


# ---------------------------------------------------------------------------
# service: attach_wall_distance_from_picks
# ---------------------------------------------------------------------------


def test_attach_wall_distance_from_picks_attaches_finite_fields() -> None:
    from naviertwin.web.service import attach_wall_distance_from_picks

    n = 8
    dataset = _make_dataset(_make_grid(n=n))
    surface = dataset.mesh.extract_surface()
    xmin = float(dataset.mesh.bounds[0])
    ids = _face_cell_ids(surface, axis=0, value=xmin)
    assert ids

    names = attach_wall_distance_from_picks(dataset, ids, prefix="wall")

    assert names  # 최소 wall_distance 는 항상 부착
    for name in names:
        assert name in dataset.field_names
        assert name in dataset.mesh.point_data
        values = np.asarray(dataset.mesh.point_data[name])
        assert values.shape == (dataset.n_points,)
        assert np.all(np.isfinite(values))


def test_attach_wall_distance_from_picks_near_face_smaller_than_far() -> None:
    """픽킹한 면 위 점은 근거리(≈0), 반대편 점은 원거리여야 한다."""
    from naviertwin.web.service import attach_wall_distance_from_picks

    n = 8
    dataset = _make_dataset(_make_grid(n=n))
    surface = dataset.mesh.extract_surface()
    xmin = float(dataset.mesh.bounds[0])
    xmax = float(dataset.mesh.bounds[1])
    ids = _face_cell_ids(surface, axis=0, value=xmin)
    assert ids

    attach_wall_distance_from_picks(dataset, ids, prefix="wall")

    distance = np.asarray(dataset.mesh.point_data["wall_distance"])
    points = np.asarray(dataset.mesh.points)
    near_mask = np.abs(points[:, 0] - xmin) < 1e-6
    far_mask = np.abs(points[:, 0] - xmax) < 1e-6
    assert near_mask.any() and far_mask.any()

    assert distance[near_mask].max() < 1e-6
    assert distance[far_mask].min() > distance[near_mask].max()


def test_attach_wall_distance_from_picks_uses_custom_prefix() -> None:
    from naviertwin.web.service import attach_wall_distance_from_picks

    dataset = _make_dataset(_make_grid())
    surface = dataset.mesh.extract_surface()
    ids = _face_cell_ids(surface, axis=0, value=0.0)

    names = attach_wall_distance_from_picks(dataset, ids, prefix="inlet")

    assert names[0] == "inlet_distance"
    assert "inlet_distance" in dataset.mesh.point_data


# ---------------------------------------------------------------------------
# service: grow_wall_selection (v5.1 후속 — seed+region growing)
# ---------------------------------------------------------------------------


def test_grow_wall_selection_expands_beyond_seed() -> None:
    from naviertwin.web.service import grow_wall_selection

    dataset = _make_dataset(_make_grid(n=8))
    surface = dataset.mesh.extract_surface()
    ids = _face_cell_ids(surface, axis=0, value=float(dataset.mesh.bounds[0]))
    seed = [ids[0]]

    grown = grow_wall_selection(dataset, seed, rings=1)

    assert grown == sorted(set(grown))  # 중복 제거·정렬
    assert set(seed).issubset(grown)
    assert len(grown) > len(seed), "1단계 확장이면 이웃이 최소 하나는 붙어야 한다"


def test_grow_wall_selection_more_rings_yields_superset() -> None:
    """rings 를 늘리면 이전 rings 결과를 항상 포함해야 한다(단조 증가)."""
    from naviertwin.web.service import grow_wall_selection

    dataset = _make_dataset(_make_grid(n=8))
    surface = dataset.mesh.extract_surface()
    ids = _face_cell_ids(surface, axis=0, value=float(dataset.mesh.bounds[0]))
    seed = [ids[len(ids) // 2]]

    grown_1 = set(grow_wall_selection(dataset, seed, rings=1))
    grown_2 = set(grow_wall_selection(dataset, seed, rings=2))
    grown_3 = set(grow_wall_selection(dataset, seed, rings=3))

    assert grown_1.issubset(grown_2)
    assert grown_2.issubset(grown_3)


def test_grow_wall_selection_zero_rings_returns_seed_only() -> None:
    from naviertwin.web.service import grow_wall_selection

    dataset = _make_dataset(_make_grid())
    surface = dataset.mesh.extract_surface()
    ids = _face_cell_ids(surface, axis=0, value=0.0)
    seed = [ids[0], ids[1]]

    grown = grow_wall_selection(dataset, seed, rings=0)

    assert grown == sorted(set(int(c) for c in seed))


def test_grow_wall_selection_stays_within_surface_cell_count() -> None:
    """넓게 확장해도 표면 셀 총수를 넘지 않는다 (인접 그래프가 유계)."""
    from naviertwin.web.service import grow_wall_selection

    dataset = _make_dataset(_make_grid(n=6))
    surface = dataset.mesh.extract_surface()
    ids = _face_cell_ids(surface, axis=0, value=float(dataset.mesh.bounds[0]))

    grown = grow_wall_selection(dataset, [ids[0]], rings=50)

    assert len(grown) <= int(surface.n_cells)
    assert all(0 <= c < int(surface.n_cells) for c in grown)


def test_grow_wall_selection_empty_raises() -> None:
    from naviertwin.web.service import grow_wall_selection

    dataset = _make_dataset(_make_grid())
    with pytest.raises(ValueError):
        grow_wall_selection(dataset, [])


def test_grow_wall_selection_out_of_range_raises() -> None:
    from naviertwin.web.service import grow_wall_selection

    dataset = _make_dataset(_make_grid())
    with pytest.raises(ValueError):
        grow_wall_selection(dataset, [10**9])


# ---------------------------------------------------------------------------
# app: trame state/controller 배선 (GL 피킹 자체는 헤드리스에서 검증 불가)
# ---------------------------------------------------------------------------


def _make_app(name: str):
    """이름이 격리된 trame server 로 앱을 생성한다 (test_web_app.py 와 동일)."""
    from trame.app import get_server

    from naviertwin.web.app import NavierTwinWebApp

    server = get_server(name, client_type="vue3")
    return NavierTwinWebApp(server=server)


def test_wall_pick_state_and_controllers_wired() -> None:
    app = _make_app("nt-test-wallpick-wiring")
    st = app.server.state
    for key in [
        "nt_bc_pick_mode",
        "nt_bc_picked_cells",
        "nt_bc_patch_name",
        "nt_bc_status",
        "nt_bc_grow_rings",
    ]:
        assert hasattr(st, key), f"missing state key: {key}"
    for cb in [
        "nt_toggle_wall_pick",
        "nt_clear_wall_pick",
        "nt_compute_wall_distance",
        "nt_grow_wall_pick",
    ]:
        assert callable(getattr(app.ctrl, cb, None)), f"missing controller: {cb}"


def test_compute_wall_distance_from_manually_picked_cells() -> None:
    """실제 피킹 대신 st.nt_bc_picked_cells 를 직접 세팅하는 헤드리스 경로."""
    app = _make_app("nt-test-wallpick-compute")
    st = app.server.state

    app.load_demo()
    assert st.nt_has_dataset is True

    surface = app.dataset.mesh.extract_surface()
    n_surface_cells = int(surface.n_cells)
    assert n_surface_cells > 0
    picked = list(range(min(5, n_surface_cells)))

    st.nt_bc_picked_cells = picked
    st.nt_bc_patch_name = "wall"

    app.compute_wall_distance_from_picks()

    assert st.nt_error == ""
    assert "wall_distance" in st.nt_fields
    assert st.nt_bc_status or st.nt_status


def test_clear_wall_pick_resets_state() -> None:
    app = _make_app("nt-test-wallpick-clear")
    st = app.server.state
    app.load_demo()

    st.nt_bc_picked_cells = [0, 1, 2]
    st.nt_bc_status = "선택된 셀 3개"

    app.clear_wall_pick()

    assert st.nt_bc_picked_cells == []
    assert st.nt_bc_status == ""


def test_compute_wall_distance_with_no_picks_sets_error_not_crash() -> None:
    app = _make_app("nt-test-wallpick-empty")
    st = app.server.state
    app.load_demo()

    st.nt_bc_picked_cells = []
    app.compute_wall_distance_from_picks()

    assert st.nt_error  # 크래시 대신 정직한 에러 안내


def test_compute_wall_distance_without_dataset_sets_error_not_crash() -> None:
    app = _make_app("nt-test-wallpick-nodataset")
    st = app.server.state

    app.compute_wall_distance_from_picks()

    assert st.nt_error


def test_toggle_wall_pick_without_dataset_fails_gracefully() -> None:
    """플로터/데이터셋이 없는 상태에서 토글해도 예외 없이 에러만 세팅되어야 한다."""
    app = _make_app("nt-test-wallpick-toggle-guard")
    st = app.server.state

    app.toggle_wall_pick_mode()  # build_ui=False → plotter=None, dataset=None

    assert st.nt_error
    assert st.nt_bc_pick_mode is False


# ---------------------------------------------------------------------------
# app: grow_wall_pick (v5.1 후속 — seed+region growing)
# ---------------------------------------------------------------------------


def test_grow_wall_pick_expands_manually_seeded_selection() -> None:
    app = _make_app("nt-test-wallpick-grow")
    st = app.server.state
    app.load_demo()
    assert st.nt_has_dataset is True

    surface = app.dataset.mesh.extract_surface()
    n_surface_cells = int(surface.n_cells)
    assert n_surface_cells > 1
    st.nt_bc_picked_cells = [0]
    st.nt_bc_grow_rings = 1

    app.grow_wall_pick()

    assert st.nt_error == ""
    assert len(st.nt_bc_picked_cells) > 1
    assert 0 in st.nt_bc_picked_cells
    assert st.nt_bc_status


def test_grow_wall_pick_without_dataset_fails_gracefully() -> None:
    app = _make_app("nt-test-wallpick-grow-nodataset")
    st = app.server.state

    app.grow_wall_pick()

    assert st.nt_error


def test_grow_wall_pick_with_no_picks_sets_error_not_crash() -> None:
    app = _make_app("nt-test-wallpick-grow-empty")
    st = app.server.state
    app.load_demo()

    st.nt_bc_picked_cells = []
    app.grow_wall_pick()

    assert st.nt_error
