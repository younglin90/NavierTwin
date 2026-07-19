"""GINO(gino) 배선 테스트 — Route 2 두 번째 전략 (점군, 고정 그래프 없음).

지키려는 계약:
    - 점군 빌더: ``case_to_pointcloud`` 의 벡터 성분 전개가 ``mesh_gnn``
      (``case_graph.case_to_graph``)/Route 1(``cases_to_grid_tensors``)의
      ``target_names`` 규약과 문자열까지 일치한다.
    - ``GINOCaseSetOperator`` 가 **점 수가 서로 다른** 케이스들을 한 모델로
      학습하고, 고정 그래프 없이 임의 점군에서도 예측한다.
    - ``GINOTwinEngine`` 이 트윈 계약(``predict``/``training_metadata``/
      ``save/load``)과 ``predict_at`` 소켓(``predict_to_mesh``)을 만족하고,
      ``varying_mesh=True`` 로 원본 케이스 메쉬 위 표시가 성립한다.
    - 앱 헤드리스 플로우: 진짜 구멍 케이스 세트 → gino 학습 → 예측이 보고
      있는 원본 케이스 메쉬에 ``twin_*`` 필드로 붙는다.

스모크 스케일 계약: 전체 파일 3분 이내 — mesh_gnn 의 karman_shapes 데모(케이스당
~6만 노드)는 GINO 의 반경 이웃 탐색·잠재 격자 FNO 비용에 비해 너무 크므로,
연구 문서(``.omc/research/route2-mesh-native-wiring.md`` §5) 지시대로 점
50~65개 소형 합성 케이스(``grid_with_hole`` 축소판)를 쓴다. epochs/은닉/잠재
해상도 모두 최소로 줄인다(save/load bit-동일 검증에 GPU 비결정성이 끼지 않게
CPU 강제).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="점군 빌더에 pyvista 가 필요합니다.")
pytest.importorskip("torch", reason="GINO 학습에 torch 가 필요합니다.")
neuralop = pytest.importorskip("neuralop", reason="gino 는 neuraloperator 가 필요합니다.")
pytest.importorskip(
    "neuralop.models", reason="gino 는 neuraloperator.models.GINO 가 필요합니다."
)

from naviertwin.core.cfd_reader.base import CFDDataset  # noqa: E402
from naviertwin.core.operator_learning.gino.gino_wrapper import (  # noqa: E402
    GINOCaseSetOperator,
    case_to_pointcloud,
    pointcloud_norm_from_cases,
)
from naviertwin.web import service  # noqa: E402

# 테스트 속도용 최소 설정 — seed 는 GINOCaseSetOperator 기본값(0).
_TINY = {
    "in_gno_radius": 0.3,
    "out_gno_radius": 0.3,
    "fno_n_modes": (2, 2, 2),
    "fno_hidden_channels": 8,
    "fno_n_layers": 2,
    "latent_resolution": 4,
    "max_epochs": 3,
    "device": "cpu",
}


def _hole_case(size: int, *, nx: int = 9, ny: int = 7) -> CFDDataset:
    """진짜 구멍(장애물 자리에 셀 없음)이 뚫린 초소형 합성 정상 케이스.

    ``size`` 마다 노드 수가 달라진다(형상 가변) — Route 2 의 존재 이유. 점
    수를 50~65 개 수준으로 눌러 GINO 스모크가 3분 예산 안에 들게 한다
    (research 문서 §5 지시 — karman_shapes 전체는 케이스당 ~6만 노드라 GINO
    반경 탐색·잠재 FNO 비용에 비해 과도하게 크다).
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
        metadata={"source": f"test_gino_hole_case_size{size}"},
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
    """구멍 크기가 다른 3케이스 — 케이스마다 n_points 가 다르다."""
    result = _hole_case_set((3, 5, 6))
    counts = {int(d.n_points) for d in result["datasets"]}
    assert len(counts) == 3, "테스트 전제: 케이스마다 노드 수가 달라야 한다"
    return result


@pytest.fixture(scope="module")
def gino_twin(hole_cases: dict) -> dict:
    """hole 케이스 세트로 학습한 gino 트윈 (module 공유 — 재학습 방지)."""
    return service.build_gino_twin_from_cases(
        hole_cases["datasets"],
        ["p", "U"],
        hole_cases["params"],
        param_names=hole_cases["param_names"],
        **_TINY,
    )


# ──────────────────────────────────────────────────────────────────────
# 점군 빌더
# ──────────────────────────────────────────────────────────────────────


def test_case_to_pointcloud_mu_broadcast_and_targets() -> None:
    """벡터 성분 전개가 mesh_gnn/Route 1 텐서화의 target_names 와 일치."""
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

    pc = case_to_pointcloud(case, mu, ["p", "U"])
    tensors = cases_to_grid_tensors(
        [case], mu.reshape(1, -1), field_names=["p", "U"], resolution=8
    )
    # 세 루트(gino/mesh_gnn/Route 1) 모두 채널 전개 규약이 문자열까지 같다.
    assert pc["target_names"] == list(tensors["meta"]["target_names"])
    assert pc["target_names"] == ["p", "U_x", "U_y", "U_z"]

    # x = [[0,1] 좌표 3 | μ 2] — μ 채널은 전 점 동일(브로드캐스트).
    assert pc["x"].shape == (grid.n_points, 5)
    for j in range(2):
        channel = pc["x"][:, 3 + j]
        assert np.allclose(channel, channel[0])
    # coords01 은 [0,1] 범위 안 (min-max 정규화).
    assert pc["coords01"].min() >= -1e-6
    assert pc["coords01"].max() <= 1.0 + 1e-6
    # y 는 물리 단위 그대로 (표준화는 GINOCaseSetOperator 내부).
    assert pc["y"].shape == (grid.n_points, 4)
    assert np.allclose(pc["y"][:, 0], pts[:, 0])


def test_pointcloud_norm_train_only_injection(hole_cases: dict) -> None:
    """train 케이스로 만든 norm 을 다른 케이스에 주입해도 μ 채널이 같은 상수를 쓴다."""
    datasets = hole_cases["datasets"]
    params = hole_cases["params"]
    norm = pointcloud_norm_from_cases(datasets[:2], params[:2])
    pc_test = case_to_pointcloud(datasets[2], params[2], ["p"], norm=norm)
    assert pc_test["norm"] is norm
    expected = (params[2, 0] - norm["mu_center"][0]) / norm["mu_scale"][0]
    assert np.allclose(pc_test["x"][:, 3], expected)


# ──────────────────────────────────────────────────────────────────────
# GINOCaseSetOperator — 크기 다른 점군 학습
# ──────────────────────────────────────────────────────────────────────


def test_gino_operator_fit_predict_varying_sizes(hole_cases: dict) -> None:
    """점 수 다른 점군 3개를 한 모델로 학습하고 각자 크기로 예측한다."""
    datasets = hole_cases["datasets"]
    params = hole_cases["params"]
    norm = pointcloud_norm_from_cases(datasets, params)
    cases = [
        case_to_pointcloud(d, params[i], ["p"], norm=norm) for i, d in enumerate(datasets)
    ]
    model = GINOCaseSetOperator(
        in_channels=int(cases[0]["x"].shape[1]),
        out_channels=1,
        **_TINY,
    )
    model.fit({"cases": cases})
    assert model.is_fitted
    for case, dataset in zip(cases, datasets):
        pred = model.predict_case(case)
        assert pred.shape == (int(dataset.n_points), 1)
        assert np.isfinite(pred).all()


def test_gino_operator_predicts_on_unseen_point_count(hole_cases: dict) -> None:
    """학습에 안 쓴 점 개수의 새 점군에서도(고정 그래프 없이) 예측한다.

    mesh_gnn 의 kNN 그래프 폴백에 해당하는 이 경로가 GINO 에서는 폴백 없이
    같은 forward 로 성립한다는 것이 Route 2 두 번째 배선의 핵심 이점이다.
    """
    datasets = hole_cases["datasets"]
    params = hole_cases["params"]
    norm = pointcloud_norm_from_cases(datasets, params)
    train_cases = [
        case_to_pointcloud(d, params[i], ["p"], norm=norm) for i, d in enumerate(datasets)
    ]
    model = GINOCaseSetOperator(in_channels=int(train_cases[0]["x"].shape[1]), out_channels=1, **_TINY)
    model.fit({"cases": train_cases})

    other = _hole_case(4)  # 학습에 쓰지 않은 크기 → 다른 점 개수
    other_case = case_to_pointcloud(other, np.asarray([4.0]), ["p"], norm=norm)
    pred = model.predict_case(other_case)
    assert pred.shape == (int(other.n_points), 1)
    assert np.isfinite(pred).all()


def test_gino_operator_rejects_bad_shapes(hole_cases: dict) -> None:
    datasets = hole_cases["datasets"]
    params = hole_cases["params"]
    case = case_to_pointcloud(datasets[0], params[0], ["p"])
    model = GINOCaseSetOperator(in_channels=99, out_channels=1, max_epochs=1, device="cpu")
    with pytest.raises(ValueError, match="case\\['x'\\]"):
        model.fit({"cases": [case]})


# ──────────────────────────────────────────────────────────────────────
# 트윈 엔진 계약
# ──────────────────────────────────────────────────────────────────────


def test_gino_engine_contract(gino_twin: dict, tmp_path) -> None:
    """predict 길이/output_fields 경계/varying_mesh/save-load bit-동일."""
    engine = gino_twin["engine"]
    meta = engine.training_metadata

    assert meta["varying_mesh"] is True
    assert meta["problem_type"] == "steady_sweep"
    assert meta["reducer"] == "gino"
    assert meta["surrogate"] == "gino_point_cloud"
    assert meta["param_names"] == ["frontal_height"]
    assert meta["param_mins"] == pytest.approx([3.0])
    assert meta["param_maxs"] == pytest.approx([6.0])
    assert not meta.get("common_grid")

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

    path = tmp_path / "gino_engine.pkl"
    engine.save(path)
    from naviertwin.core.digital_twin.gino_engine import GINOTwinEngine

    restored = GINOTwinEngine.load(path)
    assert np.array_equal(restored.predict(np.asarray([5.0])), prediction)

    with pytest.raises(ValueError, match="파라미터 차원"):
        engine.predict(np.asarray([1.0, 2.0]))


def test_predict_to_mesh_on_original_case_mesh(
    hole_cases: dict, gino_twin: dict
) -> None:
    """Route 2 존재 증명 — 예측이 구멍 뚫린 원본 케이스 메쉬 위에 그대로 붙는다."""
    engine = gino_twin["engine"]
    for i, case in enumerate(hole_cases["datasets"]):
        mu = hole_cases["params"][i]
        predicted, attached = service.predict_to_mesh(engine, mu, case)
        assert predicted.n_points == case.n_points
        assert attached == ["twin_p", "twin_U_x", "twin_U_y", "twin_U_z"]
        for name in attached:
            values = np.asarray(predicted.mesh.point_data[name])
            assert values.shape == (case.n_points,)
            assert np.isfinite(values).all()


def test_predict_at_arbitrary_point_count_no_fallback_needed(gino_twin: dict) -> None:
    """학습 케이스와 다른 점 개수의 좌표에서도 그래프 재구성 없이 예측한다.

    mesh_gnn 은 이 경우 kNN 그래프 폴백을 타지만, GINO 는 고정 그래프가 없어
    같은 코드 경로(연산자 forward)로 바로 처리한다 — predict_at 구현에
    "폴백" 분기 자체가 없다는 것을 확인한다.
    """
    engine = gino_twin["engine"]
    rng = np.random.default_rng(0)
    coords = rng.uniform(0.0, 9.0, size=(37, 3))
    coords[:, 2] = 0.0
    flat = np.asarray(engine.model.predict_at(coords, np.asarray([5.0])))
    assert flat.shape == (4 * coords.shape[0],)
    assert np.isfinite(flat).all()


def test_gino_group_split_holdout() -> None:
    """group_split=True — held-out rel-L2 유한값 + train/test 분리 보장."""
    case_set = _hole_case_set((3, 4, 5, 6))
    result = service.build_gino_twin_from_cases(
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
    built = service.build_gino_twin_from_cases(
        result["datasets"],
        "p",
        result["params"],
        param_names=result["param_names"],
        fno_n_modes=(2, 2, 2),
        fno_hidden_channels=4,
        fno_n_layers=1,
        latent_resolution=3,
        max_epochs=1,
        device="cpu",
    )

    assert built["param_names"] == ["inlet_velocity", "t"]
    assert built["engine"].training_metadata["problem_type"] == "unsteady_sweep"


# ──────────────────────────────────────────────────────────────────────
# 앱 계층 (trame state/controller — GL 없이). karman_shapes 전체 데모는
# 케이스당 ~6만 노드라 GINO 스모크 예산(3분)에 비해 과도하게 크므로, 여기서도
# 초소형 합성 케이스 세트를 ``_set_case_set`` 로 직접 주입한다(§ 파일 docstring).
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
    app._set_case_set(result, "synthetic://gino-test")


def test_app_gino_flow(hole_cases: dict) -> None:
    """진짜 구멍 케이스 세트 → gino 학습 → 원본 케이스 메쉬 위 예측.

    지금까지 Physics AI/mesh_gnn 전용이던 형상 가변 칸에 세 번째 전략이
    서는 검증이다.
    """
    app = _make_app("nt-test-gino")
    st = app.server.state

    _load_synthetic_case_set(app, hole_cases)
    assert st.nt_error == ""
    assert st.nt_case_mode is True
    # 능력 레지스트리: 형상 가변 정상 케이스 세트 → gino 카드가 켜진다.
    assert st.nt_strategy_status["gino"]["ok"] is True
    assert st.nt_strategy_status["gino"]["tier"] == "experimental"

    st.nt_model_method = "gino"
    st.nt_gino_epochs = _TINY["max_epochs"]
    st.nt_gino_hidden = _TINY["fno_hidden_channels"]
    st.nt_gino_radius = _TINY["in_gno_radius"]
    app.build_twin()
    assert st.nt_error == ""
    assert st.nt_model_ready is True
    assert st.nt_twin_ready is True
    assert "GINO" in st.nt_model_summary
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


def test_app_gino_single_case_is_rejected() -> None:
    """단일 케이스 + gino 는 케이스 세트 안내 에러를 낸다."""
    app = _make_app("nt-test-gino-single")
    st = app.server.state
    app.load_demo()  # filament 시계열 (케이스 세트 아님)
    assert st.nt_case_mode is False
    assert st.nt_strategy_status["gino"]["ok"] is False
    st.nt_model_method = "gino"
    app.build_twin()
    assert "케이스 세트" in st.nt_error
