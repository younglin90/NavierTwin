"""메쉬 GNN(mesh_gnn) 배선 테스트 — Route 2 첫 전략 (메쉬 네이티브, 재샘플 없음).

지키려는 계약:
    - 그래프 빌더: ``mesh_edge_index`` 가 점 순서를 보존한 양방향 에지를 주고,
      ``case_to_graph`` 의 벡터 성분 전개가 Route 1(``cases_to_grid_tensors``)
      의 ``target_names`` 규약과 문자열까지 일치한다.
    - ``CaseSetGNN`` 이 **노드 수가 서로 다른** 그래프들을 한 모델로 학습한다.
    - ``MeshGNNTwinEngine`` 이 트윈 계약(``predict``/``training_metadata``/
      ``save/load``)과 ``predict_at`` 소켓(``predict_to_mesh``)을 만족하고,
      ``varying_mesh=True`` 로 원본 케이스 메쉬 위 표시가 성립한다.
    - 앱 헤드리스 플로우: karman_shapes(진짜 구멍) → mesh_gnn 학습 → 예측이
      보고 있는 원본 케이스 메쉬에 ``twin_*`` 필드로 붙는다.

스모크 스케일 계약: 전체 파일 3분 이내 — epochs 소량, hidden 소형, CPU 강제
(save/load bit-동일 검증에 GPU scatter 비결정성이 끼지 않게).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="그래프 빌더에 pyvista 가 필요합니다.")
pytest.importorskip("torch", reason="GNN 학습에 torch 가 필요합니다.")
pytest.importorskip("torch_geometric", reason="mesh_gnn 은 torch_geometric 이 필요합니다.")

from naviertwin.core.cfd_reader.base import CFDDataset  # noqa: E402
from naviertwin.core.gnn.case_graph import (  # noqa: E402
    case_to_graph,
    graph_norm_from_cases,
    mesh_edge_index,
)
from naviertwin.core.gnn.gnn_surrogate.case_set_gnn import CaseSetGNN  # noqa: E402
from naviertwin.web import service  # noqa: E402

# 테스트 속도용 소형 설정 — seed 는 CaseSetGNN 기본값(0)으로 결정적이다.
_TINY = {"hidden": 32, "n_layers": 3, "max_epochs": 120, "device": "cpu"}


def _hole_case(size: int, *, nx: int = 28, ny: int = 16) -> CFDDataset:
    """진짜 구멍(장애물 자리에 셀 없음)이 뚫린 합성 정상 케이스를 만든다.

    ``size`` 마다 노드 수가 달라진다(형상 가변) — mesh_gnn 의 존재 이유.
    필드는 μ(=size)에 강하게 걸리게 해 μ 민감도(약한 방향성 검증)를 잴 수
    있게 한다.
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
        metadata={"source": f"test_hole_case_size{size}"},
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
    result = _hole_case_set((5, 7, 9))
    counts = {int(d.n_points) for d in result["datasets"]}
    assert len(counts) == 3, "테스트 전제: 케이스마다 노드 수가 달라야 한다"
    return result


@pytest.fixture(scope="module")
def hole_twin(hole_cases: dict) -> dict:
    """hole 케이스 세트로 학습한 mesh_gnn 트윈 (module 공유 — 재학습 방지)."""
    return service.build_mesh_gnn_twin_from_cases(
        hole_cases["datasets"],
        ["p", "U"],
        hole_cases["params"],
        param_names=hole_cases["param_names"],
        **_TINY,
    )


# ──────────────────────────────────────────────────────────────────────
# 그래프 빌더
# ──────────────────────────────────────────────────────────────────────


def test_mesh_edge_index_preserves_point_order() -> None:
    """에지 추출이 점 개수·순서를 보존하고, 양방향 + 범위 내 인덱스를 준다."""
    import pyvista as pv

    mesh = pv.ImageData(dimensions=(8, 8, 1)).cast_to_unstructured_grid()
    edge_index = mesh_edge_index(mesh)
    assert edge_index.dtype == np.int64
    assert edge_index.ndim == 2 and edge_index.shape[0] == 2
    assert edge_index.max() < mesh.n_points
    assert edge_index.min() >= 0
    # 양방향: (i, j) 가 있으면 (j, i) 도 있다.
    pairs = {(int(a), int(b)) for a, b in edge_index.T}
    assert all((b, a) in pairs for a, b in pairs)
    # 자기루프 없음.
    assert all(a != b for a, b in pairs)


def test_mesh_edge_index_on_mesh_with_hole() -> None:
    """진짜 구멍이 뚫린 UnstructuredGrid 에서도 점 순서 보존 계약이 성립한다."""
    case = _hole_case(7)
    edge_index = mesh_edge_index(case.mesh)
    assert edge_index.max() < case.mesh.n_points


def test_case_to_graph_mu_broadcast_and_targets() -> None:
    """벡터 성분 전개가 Route 1 텐서화의 target_names 와 일치하고 μ 는 상수 채널."""
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
    # 두 루트의 채널 전개 규약이 문자열까지 같다 — 사용자 개념 모델 일치.
    assert graph["target_names"] == list(tensors["meta"]["target_names"])
    assert graph["target_names"] == ["p", "U_x", "U_y", "U_z"]

    # x = [정규화 좌표 3 | μ 2] — μ 채널은 전 노드 동일(브로드캐스트).
    assert graph["x"].shape == (grid.n_points, 5)
    for j in range(2):
        channel = graph["x"][:, 3 + j]
        assert np.allclose(channel, channel[0])
    # y 는 물리 단위 그대로 (표준화는 CaseSetGNN 내부).
    assert graph["y"].shape == (grid.n_points, 4)
    assert np.allclose(graph["y"][:, 0], pts[:, 0])
    # edge_attr = [Δ정규화좌표(3), ‖Δ‖] — MGN 표준 상대좌표 피처.
    assert graph["edge_attr"].shape == (graph["edge_index"].shape[1], 4)


def test_graph_norm_train_only_injection(hole_cases: dict) -> None:
    """train 케이스로 만든 norm 을 다른 케이스에 주입해도 μ 채널이 같은 상수를 쓴다."""
    datasets = hole_cases["datasets"]
    params = hole_cases["params"]
    norm = graph_norm_from_cases(datasets[:2], params[:2])
    g_test = case_to_graph(datasets[2], params[2], ["p"], norm=norm)
    # 주입된 norm 이 그대로 반환된다 (재계산 없음).
    assert g_test["norm"] is norm
    expected = (params[2, 0] - norm["mu_center"][0]) / norm["mu_scale"][0]
    assert np.allclose(g_test["x"][:, 3], expected)


# ──────────────────────────────────────────────────────────────────────
# CaseSetGNN — 크기 다른 그래프 학습
# ──────────────────────────────────────────────────────────────────────


def test_case_set_gnn_fit_predict_varying_sizes(hole_cases: dict) -> None:
    """노드 수 다른 그래프 3개를 한 모델로 학습하고 각자 크기로 예측한다."""
    datasets = hole_cases["datasets"]
    params = hole_cases["params"]
    norm = graph_norm_from_cases(datasets, params)
    graphs = [
        case_to_graph(d, params[i], ["p"], norm=norm) for i, d in enumerate(datasets)
    ]
    model = CaseSetGNN(
        in_dim=int(graphs[0]["x"].shape[1]),
        out_dim=1,
        hidden=16,
        n_layers=3,
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


def test_case_set_gnn_rejects_bad_shapes(hole_cases: dict) -> None:
    datasets = hole_cases["datasets"]
    params = hole_cases["params"]
    graph = case_to_graph(datasets[0], params[0], ["p"])
    model = CaseSetGNN(in_dim=99, out_dim=1, max_epochs=1, device="cpu")
    with pytest.raises(ValueError, match="graph\\['x'\\]"):
        model.fit({"graphs": [graph]})


# ──────────────────────────────────────────────────────────────────────
# 트윈 엔진 계약
# ──────────────────────────────────────────────────────────────────────


def test_mesh_gnn_engine_contract(hole_twin: dict, tmp_path) -> None:
    """predict 길이/output_fields 경계/varying_mesh/save-load bit-동일."""
    engine = hole_twin["engine"]
    meta = engine.training_metadata

    assert meta["varying_mesh"] is True
    assert meta["problem_type"] == "steady_sweep"
    assert meta["reducer"] == "mesh_gnn"
    assert meta["surrogate"] == "case_set_gcn"
    assert meta["param_names"] == ["frontal_height"]
    assert meta["param_mins"] == pytest.approx([5.0])
    assert meta["param_maxs"] == pytest.approx([9.0])
    # 재샘플 경로 표시(common_grid)가 아니다 — Route 2 의 정체성.
    assert not meta.get("common_grid")

    # 대표(0번) 케이스 점 수 기준 field-major 벡터 + 채널 경계 정합.
    n0 = int(engine._cases[0]["points"].shape[0])
    specs = engine.model.output_fields
    assert [s["display_name"] for s in specs] == ["p", "U_x", "U_y", "U_z"]
    assert [s["field_name"] for s in specs] == ["p", "U", "U", "U"]
    assert [(s["start"], s["end"]) for s in specs] == [
        (i * n0, (i + 1) * n0) for i in range(4)
    ]
    prediction = engine.predict(np.asarray([6.0]))
    assert prediction.shape == (4 * n0,)
    assert np.isfinite(prediction).all()

    # split_multi_prediction 계약 — 채널 분해 + 벡터 크기 파생.
    parts = service.split_multi_prediction(engine, prediction)
    assert parts is not None
    assert [name for name, _ in parts] == ["p", "U_x", "U_y", "U_z", "U_mag"]

    # save/load 후 예측 bit-동일 (CPU 결정성).
    path = tmp_path / "mesh_gnn_engine.pkl"
    engine.save(path)
    from naviertwin.core.digital_twin.mesh_gnn_engine import MeshGNNTwinEngine

    restored = MeshGNNTwinEngine.load(path)
    assert np.array_equal(restored.predict(np.asarray([6.0])), prediction)

    # 파라미터 차원 검증.
    with pytest.raises(ValueError, match="파라미터 차원"):
        engine.predict(np.asarray([1.0, 2.0]))


def test_predict_to_mesh_on_original_case_mesh(
    hole_cases: dict, hole_twin: dict
) -> None:
    """Route 2 존재 증명 — 예측이 구멍 뚫린 원본 케이스 메쉬 위에 그대로 붙는다."""
    engine = hole_twin["engine"]
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


def test_training_mu_tracks_its_own_case(hole_cases: dict, hole_twin: dict) -> None:
    """약한 방향성 검증 — 학습 μ 예측이 먼 μ 예측보다 자기 참값에 가깝다.

    소표본(3케이스)이라 정량 정확도는 주장하지 않는다 — 모델이 μ 채널에
    실제로 반응한다는 방향성만 확인한다 (필드가 μ 에 비례하게 만든 데이터).
    """
    engine = hole_twin["engine"]
    case = hole_cases["datasets"][0]  # μ = 5
    coords = np.asarray(case.mesh.points, dtype=np.float64)
    truth = np.asarray(case.mesh.point_data["p"], dtype=np.float64)

    def p_prediction(mu: float) -> np.ndarray:
        flat = np.asarray(engine.model.predict_at(coords, np.asarray([mu])))
        return flat.reshape(4, -1)[0]  # 채널 0 = p

    err_own = float(np.sqrt(np.mean((p_prediction(5.0) - truth) ** 2)))
    err_far = float(np.sqrt(np.mean((p_prediction(9.0) - truth) ** 2)))
    assert err_own < err_far, (
        f"학습 μ 예측이 자기 케이스보다 먼 μ 에 더 가깝습니다 "
        f"(own={err_own:.4g}, far={err_far:.4g})"
    )


def test_predict_at_unknown_coords_uses_knn_fallback(hole_twin: dict) -> None:
    """학습 케이스와 다른 좌표 → kNN 그래프 폴백으로도 유한한 예측을 준다."""
    engine = hole_twin["engine"]
    rng = np.random.default_rng(0)
    coords = engine._cases[0]["points"] + rng.normal(0.0, 1e-3, size=(1, 3))
    flat = np.asarray(engine.model.predict_at(coords, np.asarray([6.0])))
    assert flat.shape == (4 * coords.shape[0],)
    assert np.isfinite(flat).all()


def test_mesh_gnn_group_split_holdout() -> None:
    """group_split=True — held-out rel-L2 유한값 + train/test 분리 보장."""
    case_set = _hole_case_set((4, 5, 6, 7, 8))
    result = service.build_mesh_gnn_twin_from_cases(
        case_set["datasets"],
        "p",
        case_set["params"],
        param_names=case_set["param_names"],
        hidden=16,
        n_layers=3,
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
        service.build_mesh_gnn_twin_from_cases(
            result["datasets"],
            "p",
            result["params"],
            param_names=result["param_names"],
            **_TINY,
        )


# ──────────────────────────────────────────────────────────────────────
# 앱 계층 (trame state/controller — GL 없이)
# ──────────────────────────────────────────────────────────────────────


def _make_app(name: str):
    """이름이 격리된 trame server 로 앱을 생성한다 (state 누수 방지)."""
    pytest.importorskip("trame", reason="앱 테스트에는 trame 이 필요합니다.")
    from trame.app import get_server

    from naviertwin.web.app import NavierTwinWebApp

    server = get_server(name, client_type="vue3")
    return NavierTwinWebApp(server=server)


def test_app_mesh_gnn_karman_flow() -> None:
    """karman_shapes(진짜 구멍) → mesh_gnn 학습 → 원본 케이스 메쉬 위 예측.

    지금까지 Physics AI 전용이던 형상 가변 칸에 두 번째 전략이 서는 검증이다.
    번들 데이터(demo_data/karman_caseset.npz)로 로드되므로 해석 계산은 없다.
    """
    app = _make_app("nt-test-mesh-gnn")
    st = app.server.state

    st.nt_demo_kind = "karman_shapes"
    app.load_demo()
    assert st.nt_error == ""
    assert st.nt_case_mode is True
    # 능력 레지스트리: 형상 가변 정상 케이스 세트 → mesh_gnn 카드가 켜진다.
    assert st.nt_strategy_status["mesh_gnn"]["ok"] is True
    assert st.nt_strategy_status["mesh_gnn"]["tier"] == "experimental"

    st.nt_model_method = "mesh_gnn"
    st.nt_mesh_gnn_epochs = 5  # 테스트 속도용 (배선 검증 — 정확도는 안 본다)
    st.nt_mesh_gnn_hidden = 8
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


def test_app_mesh_gnn_single_case_is_rejected() -> None:
    """단일 케이스 + mesh_gnn 은 케이스 세트 안내 에러를 낸다."""
    app = _make_app("nt-test-mesh-gnn-single")
    st = app.server.state
    app.load_demo()  # filament 시계열 (케이스 세트 아님)
    assert st.nt_case_mode is False
    assert st.nt_strategy_status["mesh_gnn"]["ok"] is False
    st.nt_model_method = "mesh_gnn"
    app.build_twin()
    assert "케이스 세트" in st.nt_error
