"""MeshGraphNets 케이스 세트(mesh_gnn_mp) 배선 테스트 — Route 2 세 번째 전략.

지키려는 계약:
    - ``CaseSetMGN`` 이 mesh_gnn(GCN, ``CaseSetGNN``)과 같은 그래프 빌더
      (``case_graph.case_to_graph``)를 그대로 쓰되, 진짜 ``edge_features``
      (Δ좌표, ‖Δ‖)를 메시지패싱에 반영하는 MeshGraphNets(Encode-Process-
      Decode)를 감싼다 — **노드 수가 서로 다른** 그래프들을 한 모델로 학습.
    - 예측은 감싼 ``MeshGraphNets.predict()`` 를 ``edge_index``/
      ``edge_features`` 오버라이드로 부른다 — 수정된(더 이상 stale 하지 않은)
      경로를 그대로 탄다.
    - ``MGNCaseSetTwinEngine`` 이 트윈 계약(``predict``/``training_metadata``/
      ``save/load``)과 ``predict_at`` 소켓(``predict_to_mesh``)을 만족하고,
      ``varying_mesh=True`` 로 원본 케이스 메쉬 위 표시가 성립한다.
    - 앱 헤드리스 플로우: 진짜 구멍 케이스 세트 → mesh_gnn_mp 학습 → 예측이
      보고 있는 원본 케이스 메쉬에 ``twin_*`` 필드로 붙는다.

스모크 스케일 계약: 전체 파일 3분 이내 — mesh_gnn 의 karman_shapes 데모
(케이스당 ~6만 노드)는 메시지패싱(edge MLP + node MLP, ``n_msgpass`` 겹)
비용이 GCN 보다 커서 이 예산에 비해 과도하게 크므로, GINO 배선(연구 문서
``.omc/research/route2-mesh-native-wiring.md`` §5) 과 같은 방식으로 점
50~65개 소형 합성 케이스(``grid_with_hole`` 축소판)를 쓴다. epochs/은닉/
메시지패싱 층 수 모두 최소로 줄이고, save/load bit-동일 검증에 GPU
비결정성이 끼지 않게 CPU 를 강제한다.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="그래프 빌더에 pyvista 가 필요합니다.")
pytest.importorskip("torch", reason="MGN 학습에 torch 가 필요합니다.")
pytest.importorskip(
    "torch_geometric", reason="mesh_gnn_mp 는 torch_geometric 이 필요합니다."
)

from naviertwin.core.cfd_reader.base import CFDDataset  # noqa: E402
from naviertwin.core.gnn.case_graph import (  # noqa: E402
    case_to_graph,
    graph_norm_from_cases,
)
from naviertwin.core.gnn.meshgraphnets.case_set_mgn import CaseSetMGN  # noqa: E402
from naviertwin.web import service  # noqa: E402

# 테스트 속도용 소형 설정 — seed 는 CaseSetMGN 기본값(0)으로 결정적이다.
_TINY = {"hidden": 16, "n_msgpass": 2, "max_epochs": 40, "device": "cpu"}


def _hole_case(size: int, *, nx: int = 9, ny: int = 7) -> CFDDataset:
    """진짜 구멍(장애물 자리에 셀 없음)이 뚫린 초소형 합성 정상 케이스.

    ``size`` 마다 노드 수가 달라진다(형상 가변) — mesh_gnn_mp 의 존재
    이유. 점 수를 50~65 개 수준으로 눌러 메시지패싱 스모크가 3분 예산 안에
    들게 한다(mesh_gnn 의 카르만 데모보다 훨씬 작다 — 연구 문서 §5 의 GINO
    배선 지시를 그대로 따른다).
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
        metadata={"source": f"test_mgn_hole_case_size{size}"},
    )


def _hole_case_set(sizes: tuple[int, ...], *, ny: int = 7) -> dict:
    """노드 수가 서로 다른 케이스 세트 (μ = 장애물 정면 높이 1개)."""
    datasets = [_hole_case(s, ny=ny) for s in sizes]
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
def mgn_twin(hole_cases: dict) -> dict:
    """hole 케이스 세트로 학습한 mesh_gnn_mp 트윈 (module 공유 — 재학습 방지)."""
    return service.build_mgn_twin_from_cases(
        hole_cases["datasets"],
        ["p", "U"],
        hole_cases["params"],
        param_names=hole_cases["param_names"],
        **_TINY,
    )


# ──────────────────────────────────────────────────────────────────────
# 그래프 빌더 재사용 확인 — case_to_graph 의 edge_attr 이 MGN 이 요구하는
# edge_feature_dim 과 일치하는지.
# ──────────────────────────────────────────────────────────────────────


def test_case_to_graph_edge_attr_matches_mgn_edge_feat() -> None:
    """벡터 성분 전개가 mesh_gnn/Route 1 의 target_names 와 일치하고, edge_attr
    이 CaseSetMGN 에 그대로 넘길 수 있는 (E, edge_feat) 모양을 가진다."""
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

    graph = case_to_graph(case, mu, ["p", "U"])
    tensors = cases_to_grid_tensors(
        [case], mu.reshape(1, -1), field_names=["p", "U"], resolution=8
    )
    assert graph["target_names"] == list(tensors["meta"]["target_names"])
    assert graph["target_names"] == ["p", "U_x", "U_y", "U_z"]

    # edge_attr = [Δ정규화좌표(3), ‖Δ‖] — MGN 이 실제로 소비하는 메시지패싱
    # 에지 피처. edge_index 개수와 정확히 맞아야 CaseSetMGN._validate_graph
    # 가 통과한다.
    edge_index = graph["edge_index"]
    edge_attr = graph["edge_attr"]
    assert edge_attr.shape == (edge_index.shape[1], 4)
    assert edge_attr.dtype == np.float32


# ──────────────────────────────────────────────────────────────────────
# CaseSetMGN — 크기 다른 그래프 학습 (메시지패싱)
# ──────────────────────────────────────────────────────────────────────


def test_case_set_mgn_fit_predict_varying_sizes(hole_cases: dict) -> None:
    """노드 수 다른 그래프 3개를 한 MeshGraphNets 로 학습하고 각자 크기로 예측한다."""
    datasets = hole_cases["datasets"]
    params = hole_cases["params"]
    norm = graph_norm_from_cases(datasets, params)
    graphs = [
        case_to_graph(d, params[i], ["p"], norm=norm) for i, d in enumerate(datasets)
    ]
    model = CaseSetMGN(
        in_dim=int(graphs[0]["x"].shape[1]),
        out_dim=1,
        edge_feat=int(graphs[0]["edge_attr"].shape[1]),
        hidden=16,
        n_msgpass=2,
        max_epochs=30,
        device="cpu",
    )
    model.fit({"graphs": graphs})
    assert model.is_fitted
    # 스모크: 학습이 실제로 진행됐다 (최종 loss < 초기 loss).
    assert model.train_losses_[-1] < model.train_losses_[0]
    for graph, dataset in zip(graphs, datasets):
        pred = model.predict_graph(graph)
        assert pred.shape == (int(dataset.n_points), 1)
        assert np.isfinite(pred).all()


def test_case_set_mgn_rejects_bad_shapes(hole_cases: dict) -> None:
    datasets = hole_cases["datasets"]
    params = hole_cases["params"]
    graph = case_to_graph(datasets[0], params[0], ["p"])
    model = CaseSetMGN(
        in_dim=99, out_dim=1, edge_feat=int(graph["edge_attr"].shape[1]),
        max_epochs=1, device="cpu",
    )
    with pytest.raises(ValueError, match="graph\\['x'\\]"):
        model.fit({"graphs": [graph]})


def test_case_set_mgn_rejects_edge_feat_mismatch(hole_cases: dict) -> None:
    """edge_attr 채널 수가 선언한 edge_feat 과 다르면 명확히 거절한다.

    ``CaseSetGNN`` 은 edge_attr 를 쓰지 않아 이 검증이 없다 — 메시지패싱을
    실제로 쓰는 CaseSetMGN 에서만 필요한 새 계약이다.
    """
    datasets = hole_cases["datasets"]
    params = hole_cases["params"]
    graph = case_to_graph(datasets[0], params[0], ["p"])
    model = CaseSetMGN(
        in_dim=int(graph["x"].shape[1]), out_dim=1, edge_feat=99,
        max_epochs=1, device="cpu",
    )
    with pytest.raises(ValueError, match="edge_attr"):
        model.fit({"graphs": [graph]})


# ──────────────────────────────────────────────────────────────────────
# 트윈 엔진 계약
# ──────────────────────────────────────────────────────────────────────


def test_mgn_engine_contract(mgn_twin: dict, tmp_path) -> None:
    """predict 길이/output_fields 경계/varying_mesh/save-load bit-동일."""
    engine = mgn_twin["engine"]
    meta = engine.training_metadata

    assert meta["varying_mesh"] is True
    assert meta["problem_type"] == "steady_sweep"
    assert meta["reducer"] == "mgn_case_set"
    assert meta["surrogate"] == "case_set_mgn"
    assert meta["param_names"] == ["frontal_height"]
    assert meta["param_mins"] == pytest.approx([3.0])
    assert meta["param_maxs"] == pytest.approx([6.0])
    # 재샘플 경로 표시(common_grid)가 아니다 — Route 2 의 정체성.
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

    # split_multi_prediction 계약 — 채널 분해 + 벡터 크기 파생.
    parts = service.split_multi_prediction(engine, prediction)
    assert parts is not None
    assert [name for name, _ in parts] == ["p", "U_x", "U_y", "U_z", "U_mag"]

    # save/load 후 예측 bit-동일 (CPU 결정성) — CaseSetMGN.__getstate__/
    # __setstate__ 가 감싼 MeshGraphNets 의 로컬 클래스(_MGN) 모델을
    # state_dict 바이트로 직렬화한다.
    path = tmp_path / "mgn_engine.pkl"
    engine.save(path)
    from naviertwin.core.digital_twin.mgn_case_set_engine import MGNCaseSetTwinEngine

    restored = MGNCaseSetTwinEngine.load(path)
    assert np.array_equal(restored.predict(np.asarray([5.0])), prediction)

    with pytest.raises(ValueError, match="파라미터 차원"):
        engine.predict(np.asarray([1.0, 2.0]))


def test_predict_to_mesh_on_original_case_mesh(
    hole_cases: dict, mgn_twin: dict
) -> None:
    """Route 2 존재 증명 — 예측이 구멍 뚫린 원본 케이스 메쉬 위에 그대로 붙는다."""
    engine = mgn_twin["engine"]
    for i, case in enumerate(hole_cases["datasets"]):
        mu = hole_cases["params"][i]
        predicted, attached = service.predict_to_mesh(engine, mu, case)
        # 재샘플 없음: 점 수(케이스마다 다름)가 그대로 보존된다.
        assert predicted.n_points == case.n_points
        assert attached == ["twin_p", "twin_U_x", "twin_U_y", "twin_U_z"]
        for name in attached:
            values = np.asarray(predicted.mesh.point_data[name])
            assert values.shape == (case.n_points,)
            assert np.isfinite(values).all()


def test_training_mu_tracks_its_own_case(hole_cases: dict, mgn_twin: dict) -> None:
    """약한 방향성 검증 — 학습 μ 예측이 먼 μ 예측보다 자기 참값에 가깝다.

    소표본(3케이스)이라 정량 정확도는 주장하지 않는다 — 모델이 μ 채널에
    실제로 반응한다는 방향성만 확인한다 (필드가 μ 에 비례하게 만든 데이터).
    """
    engine = mgn_twin["engine"]
    case = hole_cases["datasets"][0]  # μ = 3
    coords = np.asarray(case.mesh.points, dtype=np.float64)
    truth = np.asarray(case.mesh.point_data["p"], dtype=np.float64)

    def p_prediction(mu: float) -> np.ndarray:
        flat = np.asarray(engine.model.predict_at(coords, np.asarray([mu])))
        return flat.reshape(4, -1)[0]  # 채널 0 = p

    err_own = float(np.sqrt(np.mean((p_prediction(3.0) - truth) ** 2)))
    err_far = float(np.sqrt(np.mean((p_prediction(6.0) - truth) ** 2)))
    assert err_own < err_far, (
        f"학습 μ 예측이 자기 케이스보다 먼 μ 에 더 가깝습니다 "
        f"(own={err_own:.4g}, far={err_far:.4g})"
    )


def test_predict_at_unknown_coords_uses_knn_fallback(mgn_twin: dict) -> None:
    """학습 케이스와 다른 좌표 → kNN 그래프(+재계산 edge_attr) 폴백으로도 유한한
    예측을 준다."""
    engine = mgn_twin["engine"]
    rng = np.random.default_rng(0)
    coords = engine._cases[0]["points"] + rng.normal(0.0, 1e-3, size=(1, 3))
    flat = np.asarray(engine.model.predict_at(coords, np.asarray([5.0])))
    assert flat.shape == (4 * coords.shape[0],)
    assert np.isfinite(flat).all()


def test_mgn_group_split_holdout() -> None:
    """group_split=True — held-out rel-L2 유한값 + train/test 분리 보장."""
    case_set = _hole_case_set((3, 4, 5, 6, 7), ny=9)
    result = service.build_mgn_twin_from_cases(
        case_set["datasets"],
        "p",
        case_set["params"],
        param_names=case_set["param_names"],
        hidden=16,
        n_msgpass=2,
        max_epochs=20,
        device="cpu",
        group_split=True,
        val_frac=0.2,
        test_frac=0.2,
    )
    split = result["eval_split"]
    assert split["enabled"] is True
    train_set = set(split["train_idx"])
    heldout = set(split["val_idx"]) | set(split["test_idx"])
    assert heldout, "5케이스 × (0.2, 0.2) 분할이면 held-out 이 있어야 한다"
    assert train_set.isdisjoint(heldout)
    for entry in split["holdout"]:
        assert np.isfinite(entry["rel_l2"])
    # 엔진은 train 케이스만 담는다.
    assert result["engine"].training_metadata["n_cases"] == len(train_set)


def test_unsteady_case_set_is_rejected() -> None:
    """비정상 스윕은 명확히 거절한다 — 시계열을 조용히 뭉개지 않는다."""
    result = service.make_demo_case_set("sweep_unsteady", n_side=12)
    with pytest.raises(ValueError, match="미지원"):
        service.build_mgn_twin_from_cases(
            result["datasets"],
            "p",
            result["params"],
            param_names=result["param_names"],
            **_TINY,
        )


# ──────────────────────────────────────────────────────────────────────
# 앱 계층 (trame state/controller — GL 없이). karman_shapes 전체 데모는
# 메시지패싱 스모크 예산(3분)에 비해 과도하게 크므로, gino 와 같은 방식으로
# 초소형 합성 케이스 세트를 ``_set_case_set`` 로 직접 주입한다.
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
    app._set_case_set(result, "synthetic://mgn-test")


def test_app_mgn_flow(hole_cases: dict) -> None:
    """진짜 구멍 케이스 세트 → mesh_gnn_mp 학습 → 원본 케이스 메쉬 위 예측.

    지금까지 Physics AI/mesh_gnn/gino 전용이던 형상 가변 칸에 네 번째
    전략이 서는 검증이다.
    """
    app = _make_app("nt-test-mgn")
    st = app.server.state

    _load_synthetic_case_set(app, hole_cases)
    assert st.nt_error == ""
    assert st.nt_case_mode is True
    # 능력 레지스트리: 형상 가변 정상 케이스 세트 → mesh_gnn_mp 카드가 켜진다.
    assert st.nt_strategy_status["mesh_gnn_mp"]["ok"] is True
    assert st.nt_strategy_status["mesh_gnn_mp"]["tier"] == "experimental"

    st.nt_model_method = "mesh_gnn_mp"
    st.nt_mgn_epochs = _TINY["max_epochs"]
    st.nt_mgn_hidden = _TINY["hidden"]
    st.nt_mgn_msgpass = _TINY["n_msgpass"]
    app.build_twin()
    assert st.nt_error == ""
    assert st.nt_model_ready is True
    assert st.nt_twin_ready is True
    assert "메쉬 GNN" in st.nt_model_summary
    assert app.engine.training_metadata["varying_mesh"] is True

    # 중앙 파라미터(기본값)에서 예측 → 보고 있는 케이스 메쉬 위에 표시.
    viewed_points = int(app.case_datasets[0].n_points)
    app.predict()
    assert st.nt_error == ""
    assert "보고 있는 형상" in st.nt_status
    # 재샘플 없음: 뷰어 데이터셋 점 수 = 원본 케이스 점 수.
    assert int(app.dataset.n_points) == viewed_points
    twin_fields = [name for name in st.nt_fields if name.startswith("twin_")]
    assert twin_fields, "twin_* 예측 필드가 뷰어에 붙어야 한다"
    for name in twin_fields:
        values = np.asarray(app.dataset.mesh.point_data[name])
        assert values.shape == (viewed_points,)
        assert np.isfinite(values).all()

    # 학습 상태는 케이스 뷰 전환 후에도 보존된다.
    st.nt_case_index = 1
    app.select_case()
    assert st.nt_error == ""
    assert app.engine is not None


def test_app_mgn_single_case_is_rejected() -> None:
    """단일 케이스 + mesh_gnn_mp 는 케이스 세트 안내 에러를 낸다."""
    app = _make_app("nt-test-mgn-single")
    st = app.server.state
    app.load_demo()  # filament 시계열 (케이스 세트 아님)
    assert st.nt_case_mode is False
    assert st.nt_strategy_status["mesh_gnn_mp"]["ok"] is False
    st.nt_model_method = "mesh_gnn_mp"
    app.build_twin()
    assert "케이스 세트" in st.nt_error
