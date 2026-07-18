"""v5.1 후속 — 경계조건(BC) 값 입력 테스트.

BC 값(속도/압력/온도 등)은 점마다 다른 공간 필드가 아니라 patch 를 통째로
설명하는 메타데이터이므로 ``dataset.metadata["boundary_conditions"]`` 에
저장된다 (:mod:`naviertwin.web.service` 의 ``attach_boundary_condition_values``
/ ``list_boundary_patches`` 참조). 이 테스트는 서비스 계층의 순수 함수와,
app 레벨(:mod:`naviertwin.web.app`) 상태/콜백 배선을 검증한다.
"""

from __future__ import annotations

import pytest

pv = pytest.importorskip("pyvista", reason="pyvista 가 설치되어 있지 않음")
pytest.importorskip("trame", reason="app 레벨 테스트에는 trame 이 필요합니다.")


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _make_dataset(*, boundary_patches: dict | None = None):
    """작은 합성 CFDDataset 을 생성한다 (test_wall_picking.py 와 동일 스타일)."""
    import numpy as np

    from naviertwin.core.cfd_reader.base import CFDDataset

    mesh = pv.ImageData(dimensions=(4, 4, 4), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    ug = mesh.cast_to_unstructured_grid()
    ug.point_data["p"] = np.random.rand(ug.n_points).astype(np.float32)
    metadata: dict = {"reader": "synthetic"}
    if boundary_patches is not None:
        metadata["boundary_patches"] = boundary_patches
    return CFDDataset(mesh=ug, time_steps=[0.0], field_names=["p"], metadata=metadata)


# ---------------------------------------------------------------------------
# service: attach_boundary_condition_values
# ---------------------------------------------------------------------------


def test_attach_boundary_condition_values_stores_into_metadata() -> None:
    from naviertwin.web.service import attach_boundary_condition_values

    dataset = _make_dataset()
    names = attach_boundary_condition_values(
        dataset, "inlet", {"u": 1.0, "v": 0.0, "w": 0.0}, cell_ids=[1, 2, 3]
    )

    assert names == ["inlet"]
    stored = dataset.metadata["boundary_conditions"]["inlet"]
    assert stored["values"] == {"u": 1.0, "v": 0.0, "w": 0.0}
    assert stored["cell_ids"] == [1, 2, 3]


def test_attach_boundary_condition_values_without_cell_ids() -> None:
    from naviertwin.web.service import attach_boundary_condition_values

    dataset = _make_dataset()
    attach_boundary_condition_values(dataset, "outlet", {"p": 0.0})

    stored = dataset.metadata["boundary_conditions"]["outlet"]
    assert stored["values"] == {"p": 0.0}
    assert stored["cell_ids"] is None


def test_attach_boundary_condition_values_accumulates_multiple_patches() -> None:
    from naviertwin.web.service import attach_boundary_condition_values

    dataset = _make_dataset()
    attach_boundary_condition_values(dataset, "inlet", {"u": 1.0})
    names = attach_boundary_condition_values(dataset, "outlet", {"p": 0.0})

    assert set(names) == {"inlet", "outlet"}


def test_attach_boundary_condition_values_empty_patch_name_raises() -> None:
    from naviertwin.web.service import attach_boundary_condition_values

    dataset = _make_dataset()
    with pytest.raises(ValueError):
        attach_boundary_condition_values(dataset, "", {"u": 1.0})


def test_attach_boundary_condition_values_empty_values_raises() -> None:
    from naviertwin.web.service import attach_boundary_condition_values

    dataset = _make_dataset()
    with pytest.raises(ValueError):
        attach_boundary_condition_values(dataset, "inlet", {})


# ---------------------------------------------------------------------------
# service: list_boundary_patches
# ---------------------------------------------------------------------------


def test_list_boundary_patches_merges_file_and_picked_sources() -> None:
    from naviertwin.web.service import attach_boundary_condition_values, list_boundary_patches

    dataset = _make_dataset(
        boundary_patches={"inlet": {"n_cells": 10, "n_points": 20}}
    )
    attach_boundary_condition_values(dataset, "wall", {"u": 0.0, "v": 0.0, "w": 0.0}, cell_ids=[0, 1])

    patches = list_boundary_patches(dataset)
    by_name = {p["name"]: p for p in patches}

    assert set(by_name) == {"inlet", "wall"}
    assert by_name["inlet"]["source"] == "file"
    assert by_name["inlet"]["n_cells"] == 10
    assert by_name["inlet"]["bc_values"] is None

    assert by_name["wall"]["source"] == "picked"
    assert by_name["wall"]["n_cells"] == 2
    assert by_name["wall"]["bc_values"] == {"u": 0.0, "v": 0.0, "w": 0.0}


def test_list_boundary_patches_same_name_merges_file_meta_and_bc_values() -> None:
    from naviertwin.web.service import attach_boundary_condition_values, list_boundary_patches

    dataset = _make_dataset(
        boundary_patches={"inlet": {"n_cells": 10, "n_points": 20}}
    )
    attach_boundary_condition_values(dataset, "inlet", {"u": 2.0})

    patches = list_boundary_patches(dataset)
    assert len(patches) == 1
    assert patches[0]["name"] == "inlet"
    assert patches[0]["source"] == "file"
    assert patches[0]["n_cells"] == 10
    assert patches[0]["bc_values"] == {"u": 2.0}


def test_list_boundary_patches_empty_dataset_returns_empty_list() -> None:
    from naviertwin.web.service import list_boundary_patches

    dataset = _make_dataset()
    assert list_boundary_patches(dataset) == []


# ---------------------------------------------------------------------------
# app: trame state/controller 배선
# ---------------------------------------------------------------------------


def _make_app(name: str):
    """이름이 격리된 trame server 로 앱을 생성한다 (test_wall_picking.py 와 동일)."""
    from trame.app import get_server

    from naviertwin.web.app import NavierTwinWebApp

    server = get_server(name, client_type="vue3")
    return NavierTwinWebApp(server=server)


def test_bc_value_state_and_controller_wired() -> None:
    app = _make_app("nt-test-bcvalue-wiring")
    st = app.server.state
    for key in [
        "nt_bc_value_type",
        "nt_bc_value_x",
        "nt_bc_value_y",
        "nt_bc_value_z",
        "nt_bc_patches",
    ]:
        assert hasattr(st, key), f"missing state key: {key}"
    assert callable(getattr(app.ctrl, "nt_attach_bc", None))


def test_refresh_bc_patches_called_on_load_demo() -> None:
    app = _make_app("nt-test-bcvalue-refresh-on-load")
    st = app.server.state

    app.load_demo()

    assert st.nt_has_dataset is True
    # 합성 데모에는 OpenFOAM 경계 패치가 없으므로 빈 목록 — None/에러가 아니다.
    assert st.nt_bc_patches == []


def test_attach_boundary_condition_velocity() -> None:
    app = _make_app("nt-test-bcvalue-velocity")
    st = app.server.state
    app.load_demo()

    st.nt_bc_patch_name = "inlet"
    st.nt_bc_value_type = "velocity"
    st.nt_bc_value_x = 5.0
    st.nt_bc_value_y = 1.0
    st.nt_bc_value_z = 0.0

    app.attach_boundary_condition()

    assert st.nt_error == ""
    by_name = {p["name"]: p for p in st.nt_bc_patches}
    assert by_name["inlet"]["bc_values"] == {"u": 5.0, "v": 1.0, "w": 0.0}


def test_attach_boundary_condition_pressure() -> None:
    app = _make_app("nt-test-bcvalue-pressure")
    st = app.server.state
    app.load_demo()

    st.nt_bc_patch_name = "outlet"
    st.nt_bc_value_type = "pressure"
    st.nt_bc_value_x = 101325.0

    app.attach_boundary_condition()

    assert st.nt_error == ""
    by_name = {p["name"]: p for p in st.nt_bc_patches}
    assert by_name["outlet"]["bc_values"] == {"p": 101325.0}


def test_attach_boundary_condition_temperature() -> None:
    app = _make_app("nt-test-bcvalue-temperature")
    st = app.server.state
    app.load_demo()

    st.nt_bc_patch_name = "wall"
    st.nt_bc_value_type = "temperature"
    st.nt_bc_value_x = 300.0

    app.attach_boundary_condition()

    assert st.nt_error == ""
    by_name = {p["name"]: p for p in st.nt_bc_patches}
    assert by_name["wall"]["bc_values"] == {"T": 300.0}


def test_attach_boundary_condition_custom() -> None:
    app = _make_app("nt-test-bcvalue-custom")
    st = app.server.state
    app.load_demo()

    st.nt_bc_patch_name = "farfield"
    st.nt_bc_value_type = "custom"
    st.nt_bc_value_x = 42.0

    app.attach_boundary_condition()

    assert st.nt_error == ""
    by_name = {p["name"]: p for p in st.nt_bc_patches}
    assert by_name["farfield"]["bc_values"] == {"value": 42.0}


def test_attach_boundary_condition_without_dataset_sets_error_not_crash() -> None:
    app = _make_app("nt-test-bcvalue-nodataset")
    st = app.server.state

    app.attach_boundary_condition()

    assert st.nt_error


def test_attach_boundary_condition_without_patch_name_sets_error() -> None:
    app = _make_app("nt-test-bcvalue-nopatchname")
    st = app.server.state
    app.load_demo()

    st.nt_bc_patch_name = ""
    app.attach_boundary_condition()

    assert st.nt_error


def test_attach_boundary_condition_includes_picked_cell_ids() -> None:
    app = _make_app("nt-test-bcvalue-pickedcells")
    st = app.server.state
    app.load_demo()

    st.nt_bc_patch_name = "wall"
    st.nt_bc_value_type = "velocity"
    st.nt_bc_value_x = 0.0
    st.nt_bc_picked_cells = [0, 1, 2]

    app.attach_boundary_condition()

    assert st.nt_error == ""
    stored = app.dataset.metadata["boundary_conditions"]["wall"]
    assert stored["cell_ids"] == [0, 1, 2]
