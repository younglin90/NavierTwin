"""DeepONet(deeponet) 배선 테스트 — operator 전략 3번째 backend (순수 데이터, PDE 잔차 없음).

지키려는 계약:
    - ``DeepONetCaseSetOperator`` 가 branch(운전조건 μ)·trunk(쿼리 좌표) 두
      MLP 만으로 케이스 세트를 학습한다 — 케이스마다 메쉬가 같으면(identical
      mesh) 배치 경로, 다르면(형상 가변) 케이스별 루프 경로를 탄다.
    - ``case_to_deeponet_sample`` 의 벡터 성분 전개가 ``mesh_gnn``/``gino``와
      ``target_names`` 문자열까지 일치한다(``gino_wrapper.case_to_pointcloud``
      를 그대로 위임하므로).
    - ``DeepONetTwinEngine`` 이 트윈 계약(``predict``/``training_metadata``/
      ``save/load``)과 ``predict_at`` 소켓(``predict_to_mesh``)을 만족하고,
      ``varying_mesh=True`` 로 원본 케이스 메쉬 위 표시가 성립한다 — trunk 가
      좌표만 받으므로 mesh_gnn/GINO 의 "케이스 매칭" 분기 없이 항상 같은
      경로로 임의 좌표 예측이 된다.
    - 앱 헤드리스 플로우: 진짜 구멍 케이스 세트 → deeponet 학습 → 예측이 보고
      있는 원본 케이스 메쉬에 ``twin_*`` 필드로 붙는다.
    - 단일 케이스는 능력 판정(``strategies._check``)에서 명시적으로 거절된다.

스모크 스케일 계약: 전체 파일 3분 이내 — ``gino`` 테스트와 같은 원칙으로 점
50~65개 소형 합성 케이스(``grid_with_hole`` 축소판)를 쓴다. epochs/은닉/잠재
폭 모두 최소로 줄인다(save/load bit-동일 검증에 GPU 비결정성이 끼지 않게 CPU
강제).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="점군 빌더에 pyvista 가 필요합니다.")
pytest.importorskip("torch", reason="DeepONet 학습에 torch 가 필요합니다.")

from naviertwin.core.cfd_reader.base import CFDDataset  # noqa: E402
from naviertwin.core.operator_learning.deeponet.case_set_deeponet import (  # noqa: E402
    DeepONetCaseSetOperator,
    case_to_deeponet_sample,
    deeponet_norm_from_cases,
)
from naviertwin.web import service  # noqa: E402

# 테스트 속도용 최소 설정 — seed 는 DeepONetCaseSetOperator 기본값(0).
_TINY = {
    "hidden": 16,
    "latent": 8,
    "n_branch_layers": 2,
    "n_trunk_layers": 2,
    "max_epochs": 5,
    "device": "cpu",
}


def _hole_case(size: int, *, nx: int = 9, ny: int = 7) -> CFDDataset:
    """진짜 구멍(장애물 자리에 셀 없음)이 뚫린 초소형 합성 정상 케이스.

    ``size`` 마다 노드 수가 달라진다(형상 가변) — ``gino``/``mesh_gnn`` 테스트와
    같은 축소판 합성 데이터를 재사용해 스모크 예산(3분) 안에 든다.
    """
    from naviertwin.core.solvers.lbm_obstacle_2d import shape_mask
    from naviertwin.web.demo_karman import grid_with_hole

    solid = shape_mask(nx, ny, kind="circle", size=size)
    mesh, _keep = grid_with_hole(solid)
    pts = np.asarray(mesh.points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    mu = float(size)
    mesh.point_data["p"] = mu * (x / nx + y / ny) + 0.1 * np.sin(x / 3.0)
    mesh.point_data["U"] = np.column_stack(
        [mu * np.cos(y / ny), -mu * np.sin(x / nx), np.zeros_like(x)]
    )
    return CFDDataset(
        mesh=mesh,
        time_steps=[0.0],
        field_names=["p", "U"],
        metadata={"source": f"test_deeponet_hole_case_size{size}"},
    )


def _flat_grid_case(mu_value: float, *, nx: int = 9, ny: int = 7) -> CFDDataset:
    """구멍 없는 균일 격자 케이스 — 모든 케이스가 정확히 같은 좌표를 공유한다.

    identical-mesh 배치 학습 경로(:meth:`DeepONetCaseSetOperator.fit` 의
    ``identical_mesh=True`` 분기)를 확인하는 용도.
    """
    import pyvista as pv

    grid = pv.ImageData(dimensions=(nx, ny, 1)).cast_to_unstructured_grid()
    pts = np.asarray(grid.points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    grid.point_data["p"] = mu_value * (x / nx + y / ny)
    return CFDDataset(
        mesh=grid,
        time_steps=[0.0],
        field_names=["p"],
        metadata={"source": f"test_deeponet_flat_mu{mu_value}"},
    )


def _hole_case_set(sizes: tuple[int, ...]) -> dict:
    """노드 수가 서로 다른 케이스 세트 (μ = 장애물 정면 높이 1개)."""
    datasets = [_hole_case(s) for s in sizes]
    return {
        "datasets": datasets,
        "params": np.asarray(sizes, dtype=np.float64).reshape(-1, 1),
        "param_names": ["frontal_height"],
    }


@pytest.fixture(scope="module")
def hole_cases() -> dict:
    """구멍 크기가 다른 3케이스 — 케이스마다 n_points 가 다르다(형상 가변)."""
    result = _hole_case_set((3, 5, 6))
    counts = {int(d.n_points) for d in result["datasets"]}
    assert len(counts) == 3, "테스트 전제: 케이스마다 노드 수가 달라야 한다"
    return result


@pytest.fixture(scope="module")
def deeponet_twin(hole_cases: dict) -> dict:
    """hole 케이스 세트로 학습한 deeponet 트윈 (module 공유 — 재학습 방지)."""
    return service.build_deeponet_twin_from_cases(
        hole_cases["datasets"],
        ["p", "U"],
        hole_cases["params"],
        param_names=hole_cases["param_names"],
        **_TINY,
    )


# ──────────────────────────────────────────────────────────────────────
# 샘플 빌더
# ──────────────────────────────────────────────────────────────────────


def test_case_to_deeponet_sample_targets_match_gino_and_mesh_gnn() -> None:
    """target_names 전개가 gino/mesh_gnn/Route 1 텐서화와 문자열까지 일치."""
    import pyvista as pv

    from naviertwin.core.operator_learning.fno.case_tensorizer import (
        cases_to_grid_tensors,
    )

    grid = pv.ImageData(dimensions=(8, 8, 1)).cast_to_unstructured_grid()
    pts = np.asarray(grid.points)
    grid.point_data["p"] = pts[:, 0]
    grid.point_data["U"] = np.column_stack(
        [pts[:, 0], pts[:, 1], np.zeros(grid.n_points)]
    )
    case = CFDDataset(mesh=grid, time_steps=[0.0], field_names=["p", "U"], metadata={})
    mu = np.asarray([3.5, -1.0])

    sample = case_to_deeponet_sample(case, mu, ["p", "U"])
    tensors = cases_to_grid_tensors(
        [case], mu.reshape(1, -1), field_names=["p", "U"], resolution=8
    )
    assert sample["target_names"] == list(tensors["meta"]["target_names"])
    assert sample["target_names"] == ["p", "U_x", "U_y", "U_z"]
    assert sample["coords01"].min() >= -1e-6
    assert sample["coords01"].max() <= 1.0 + 1e-6
    assert sample["y"].shape == (grid.n_points, 4)
    assert np.allclose(sample["y"][:, 0], pts[:, 0])


# ──────────────────────────────────────────────────────────────────────
# DeepONetCaseSetOperator — 크기 다른 점군 학습 (형상 가변, 케이스별 trunk 루프)
# ──────────────────────────────────────────────────────────────────────


def test_deeponet_operator_fit_predict_varying_sizes(hole_cases: dict) -> None:
    """점 수 다른 케이스 3개를 한 모델(branch=μ, trunk=좌표)로 학습한다."""
    datasets = hole_cases["datasets"]
    params = hole_cases["params"]
    norm = deeponet_norm_from_cases(datasets, params)
    samples = [
        case_to_deeponet_sample(d, params[i], ["p"], norm=norm)
        for i, d in enumerate(datasets)
    ]
    mu_center = np.asarray(norm["mu_center"])
    mu_scale = np.asarray(norm["mu_scale"])
    cases = [
        {
            "mu": ((params[i] - mu_center) / mu_scale).astype(np.float32),
            "coords01": s["coords01"],
            "y": s["y"],
        }
        for i, s in enumerate(samples)
    ]
    model = DeepONetCaseSetOperator(branch_in=1, n_channels=1, **_TINY)
    model.fit({"cases": cases})
    assert model.is_fitted
    assert model.identical_mesh is False, "구멍 케이스는 점 수가 달라 형상 가변이어야 한다"
    for case, dataset in zip(cases, datasets):
        pred = model.predict_case(case)
        assert pred.shape == (int(dataset.n_points), 1)
        assert np.isfinite(pred).all()


def test_deeponet_operator_identical_mesh_uses_batch_path() -> None:
    """모든 케이스가 같은 좌표를 공유하면 identical_mesh=True(배치 경로)."""
    datasets = [_flat_grid_case(mu) for mu in (1.0, 2.0, 3.0, 4.0)]
    params = np.asarray([1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
    norm = deeponet_norm_from_cases(datasets, params)
    samples = [
        case_to_deeponet_sample(d, params[i], ["p"], norm=norm)
        for i, d in enumerate(datasets)
    ]
    mu_center = np.asarray(norm["mu_center"])
    mu_scale = np.asarray(norm["mu_scale"])
    cases = [
        {
            "mu": ((params[i] - mu_center) / mu_scale).astype(np.float32),
            "coords01": s["coords01"],
            "y": s["y"],
        }
        for i, s in enumerate(samples)
    ]
    model = DeepONetCaseSetOperator(branch_in=1, n_channels=1, **_TINY)
    model.fit({"cases": cases})
    assert model.is_fitted
    assert model.identical_mesh is True
    pred = model.predict_case(cases[0])
    assert pred.shape == (int(datasets[0].n_points), 1)
    assert np.isfinite(pred).all()


def test_deeponet_operator_predicts_on_unseen_point_count(hole_cases: dict) -> None:
    """학습에 안 쓴 점 개수의 새 좌표에서도(케이스 매칭 없이) 바로 예측한다.

    DeepONet 은 점 단위 입력 피처가 없어 mesh_gnn 의 kNN 폴백이나 GINO 의
    케이스 매칭 분기 자체가 없다는 것이 이 백엔드의 핵심 이점이다.
    """
    datasets = hole_cases["datasets"]
    params = hole_cases["params"]
    norm = deeponet_norm_from_cases(datasets, params)
    mu_center = np.asarray(norm["mu_center"])
    mu_scale = np.asarray(norm["mu_scale"])
    train_cases = [
        {
            "mu": ((params[i] - mu_center) / mu_scale).astype(np.float32),
            "coords01": case_to_deeponet_sample(d, params[i], ["p"], norm=norm)["coords01"],
            "y": case_to_deeponet_sample(d, params[i], ["p"], norm=norm)["y"],
        }
        for i, d in enumerate(datasets)
    ]
    model = DeepONetCaseSetOperator(branch_in=1, n_channels=1, **_TINY)
    model.fit({"cases": train_cases})

    rng = np.random.default_rng(0)
    coords01 = rng.uniform(0.0, 1.0, size=(41, 3)).astype(np.float32)
    coords01[:, 2] = 0.0
    other_mu = ((np.asarray([4.0]) - mu_center) / mu_scale).astype(np.float32)
    pred = model.predict_case({"mu": other_mu, "coords01": coords01})
    assert pred.shape == (41, 1)
    assert np.isfinite(pred).all()


def test_deeponet_operator_rejects_bad_shapes(hole_cases: dict) -> None:
    datasets = hole_cases["datasets"]
    params = hole_cases["params"]
    sample = case_to_deeponet_sample(datasets[0], params[0], ["p"])
    model = DeepONetCaseSetOperator(branch_in=99, n_channels=1, max_epochs=1, device="cpu")
    with pytest.raises(ValueError, match="case\\['mu'\\]"):
        model.fit(
            {
                "cases": [
                    {
                        "mu": np.asarray(params[0], dtype=np.float32),
                        "coords01": sample["coords01"],
                        "y": sample["y"],
                    }
                ]
            }
        )


# ──────────────────────────────────────────────────────────────────────
# 트윈 엔진 계약
# ──────────────────────────────────────────────────────────────────────


def test_deeponet_engine_contract(deeponet_twin: dict, tmp_path) -> None:
    """predict 길이/output_fields 경계/varying_mesh/save-load bit-동일."""
    engine = deeponet_twin["engine"]
    meta = engine.training_metadata

    assert meta["varying_mesh"] is True
    assert meta["problem_type"] == "steady_sweep"
    assert meta["reducer"] == "deeponet"
    assert meta["surrogate"] == "deeponet_branch_trunk"
    assert meta["param_names"] == ["frontal_height"]
    assert meta["param_mins"] == pytest.approx([3.0])
    assert meta["param_maxs"] == pytest.approx([6.0])
    assert meta["identical_mesh_training"] is False

    n0 = int(engine._cases[0]["points"].shape[0])
    specs = engine.model.output_fields
    assert [s["display_name"] for s in specs] == ["p", "U_x", "U_y", "U_z"]
    assert [s["field_name"] for s in specs] == ["p", "U", "U", "U"]
    assert [(s["start"], s["end"]) for s in specs] == [
        (i * n0, (i + 1) * n0) for i in range(4)
    ]
    prediction = engine.predict(np.asarray([5.0]))
    assert prediction.shape == (4 * n0,)
    assert np.isfinite(prediction).all()

    parts = service.split_multi_prediction(engine, prediction)
    assert parts is not None
    assert [name for name, _ in parts] == ["p", "U_x", "U_y", "U_z", "U_mag"]

    path = tmp_path / "deeponet_engine.pkl"
    engine.save(path)
    from naviertwin.core.digital_twin.deeponet_engine import DeepONetTwinEngine

    restored = DeepONetTwinEngine.load(path)
    assert np.array_equal(restored.predict(np.asarray([5.0])), prediction)

    with pytest.raises(ValueError, match="파라미터 차원"):
        engine.predict(np.asarray([1.0, 2.0]))


def test_predict_to_mesh_on_original_case_mesh(
    hole_cases: dict, deeponet_twin: dict
) -> None:
    """예측이 구멍 뚫린 원본 케이스 메쉬 위에 재샘플 없이 그대로 붙는다."""
    engine = deeponet_twin["engine"]
    for i, case in enumerate(hole_cases["datasets"]):
        mu = hole_cases["params"][i]
        predicted, attached = service.predict_to_mesh(engine, mu, case)
        assert predicted.n_points == case.n_points
        assert attached == ["twin_p", "twin_U_x", "twin_U_y", "twin_U_z"]
        for name in attached:
            values = np.asarray(predicted.mesh.point_data[name])
            assert values.shape == (case.n_points,)
            assert np.isfinite(values).all()


def test_predict_at_arbitrary_point_count_no_case_matching(deeponet_twin: dict) -> None:
    """학습 케이스와 다른 점 개수의 좌표에서도 케이스 매칭 없이 예측한다."""
    engine = deeponet_twin["engine"]
    rng = np.random.default_rng(0)
    coords = rng.uniform(0.0, 9.0, size=(37, 3))
    coords[:, 2] = 0.0
    flat = np.asarray(engine.model.predict_at(coords, np.asarray([5.0])))
    assert flat.shape == (4 * coords.shape[0],)
    assert np.isfinite(flat).all()


def test_deeponet_group_split_holdout() -> None:
    """group_split=True — held-out rel-L2 유한값 + train/test 분리 보장."""
    case_set = _hole_case_set((3, 4, 5, 6))
    result = service.build_deeponet_twin_from_cases(
        case_set["datasets"],
        "p",
        case_set["params"],
        param_names=case_set["param_names"],
        **_TINY,
        group_split=True,
        val_frac=0.25,
        test_frac=0.25,
    )
    split = result["eval_split"]
    assert split["enabled"] is True
    train_set = set(split["train_idx"])
    heldout = set(split["val_idx"]) | set(split["test_idx"])
    assert heldout, "4케이스 × (0.25, 0.25) 분할이면 held-out 이 있어야 한다"
    assert train_set.isdisjoint(heldout)
    for entry in split["holdout"]:
        assert np.isfinite(entry["rel_l2"])
    assert result["engine"].training_metadata["n_cases"] == len(train_set)


def test_unsteady_case_set_expands_time_parameter() -> None:
    result = service.make_demo_case_set("sweep_unsteady", n_side=12)
    built = service.build_deeponet_twin_from_cases(
        result["datasets"],
        "p",
        result["params"],
        param_names=result["param_names"],
        hidden=8,
        latent=4,
        n_branch_layers=2,
        n_trunk_layers=2,
        max_epochs=1,
        device="cpu",
    )

    assert built["param_names"] == ["inlet_velocity", "t"]
    assert built["engine"].training_metadata["problem_type"] == "unsteady_sweep"


# ──────────────────────────────────────────────────────────────────────
# 앱 계층 (trame state/controller — GL 없이).
# ──────────────────────────────────────────────────────────────────────


def _make_app(name: str):
    """이름이 격리된 trame server 로 앱을 생성한다 (state 누수 방지)."""
    pytest.importorskip("trame", reason="앱 테스트에는 trame 이 필요합니다.")
    from trame.app import get_server

    from naviertwin.web.app import NavierTwinWebApp

    server = get_server(name, client_type="vue3")
    return NavierTwinWebApp(server=server)


def _load_synthetic_case_set(app, hole_cases: dict) -> None:
    """초소형 hole 케이스 세트를 :meth:`_set_case_set` 로 직접 주입한다."""
    datasets = hole_cases["datasets"]
    result = {
        "datasets": datasets,
        "params": hole_cases["params"],
        "param_names": hole_cases["param_names"],
        "case_names": [f"case_{i}" for i in range(len(datasets))],
        "params_source": "synthetic_test",
        "resampled": False,
        "grid_summary": "",
    }
    app._set_case_set(result, "synthetic://deeponet-test")


def test_app_deeponet_flow(hole_cases: dict) -> None:
    """진짜 구멍 케이스 세트 → deeponet 학습 → 원본 케이스 메쉬 위 예측.

    지금까지 GeometryFNO 전용이던 operator 카드에 세 번째 backend 로 서는
    검증이다 — mesh_gnn/GINO/mesh_gnn_mp 다음, 네 번째 형상 가변 전략.
    """
    app = _make_app("nt-test-deeponet")
    st = app.server.state

    _load_synthetic_case_set(app, hole_cases)
    assert st.nt_error == ""
    assert st.nt_case_mode is True
    # 능력 레지스트리: 형상 가변 정상 케이스 세트 → deeponet 카드가 켜진다.
    assert st.nt_strategy_status["deeponet"]["ok"] is True
    assert st.nt_strategy_status["deeponet"]["tier"] == "experimental"

    st.nt_model_method = "deeponet"
    st.nt_deeponet_epochs = _TINY["max_epochs"]
    st.nt_deeponet_hidden = _TINY["hidden"]
    st.nt_deeponet_latent = _TINY["latent"]
    st.nt_deeponet_layers = _TINY["n_branch_layers"]
    app.build_twin()
    assert st.nt_error == ""
    assert st.nt_model_ready is True
    assert st.nt_twin_ready is True
    assert "DeepONet" in st.nt_model_summary
    assert app.engine.training_metadata["varying_mesh"] is True

    viewed_points = int(app.case_datasets[0].n_points)
    app.predict()
    assert st.nt_error == ""
    assert "보고 있는 형상" in st.nt_status
    assert int(app.dataset.n_points) == viewed_points
    twin_fields = [name for name in st.nt_fields if name.startswith("twin_")]
    assert twin_fields, "twin_* 예측 필드가 뷰어에 붙어야 한다"
    for name in twin_fields:
        values = np.asarray(app.dataset.mesh.point_data[name])
        assert values.shape == (viewed_points,)
        assert np.isfinite(values).all()

    st.nt_case_index = 1
    app.select_case()
    assert st.nt_error == ""
    assert app.engine is not None


def test_app_deeponet_single_case_is_rejected() -> None:
    """단일 케이스 + deeponet 은 케이스 세트 안내 에러를 낸다."""
    app = _make_app("nt-test-deeponet-single")
    st = app.server.state
    app.load_demo()  # filament 시계열 (케이스 세트 아님)
    assert st.nt_case_mode is False
    assert st.nt_strategy_status["deeponet"]["ok"] is False
    st.nt_model_method = "deeponet"
    app.build_twin()
    assert "케이스 세트" in st.nt_error
