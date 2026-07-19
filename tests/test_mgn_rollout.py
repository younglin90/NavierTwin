"""MeshGraphNets 롤아웃(mesh_gnn_rollout) 배선 테스트 — 진짜 자기회귀 시간 롤아웃.

지키려는 계약:
    - :func:`~naviertwin.core.gnn.meshgraphnets.rollout_dataset.
      case_to_rollout_trajectory` 가 단일 케이스(다중 타임스텝)를 원본
      ``MeshGraphNets.fit()`` 이 요구하는 ``trajectories`` (n_traj=1, T+1, N, C)
      + 공유 ``edge_index``/``edge_features`` 로 정확히 바꾼다.
    - ``service.build_mgn_rollout_twin`` 이 원본 ``MeshGraphNets`` 를 그대로
      학습한다(``fit()`` 호출) — ``mesh_gnn_mp``(CaseSetMGN)와 달리 "1스텝
      가짜 트레젝토리" 가 아니라 진짜 여러 타임스텝을 학습한다.
    - ``MGNRolloutTwinEngine`` 이 시간 t 를 스텝 수로 바꿔 롤아웃하고, 학습
      구간 밖(미래) t 도 유한한 값으로 외삽하며, 롤아웃 캐시를 재사용해
      먼 미래를 다시 요청해도 처음부터 다시 굴리지 않는다.
    - ``training_metadata["varying_mesh"]=True`` 라 ``predict_to_mesh`` 가
      원본 케이스 메쉬 위에 재샘플 없이 표시한다.
    - 전략 레지스트리에서 ``mesh_gnn_rollout`` 은 다른 7개와 정반대 —
      케이스 세트는 거절하고 단일 케이스 시계열만 받는다
      (``supports_case_sets=False``).

스모크 스케일 계약: 전체 파일 3분 이내 — 6×6 격자(36 점) × 5 타임스텝의
초소형 합성 비정상 케이스를 쓴다. epochs/은닉/메시지패싱 층 수 모두 최소로
줄이고, save/load bit-동일 검증에 GPU 비결정성이 끼지 않게 CPU 를 강제한다.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="그래프 빌더에 pyvista 가 필요합니다.")
pytest.importorskip("torch", reason="MGN 학습에 torch 가 필요합니다.")
pytest.importorskip(
    "torch_geometric", reason="mesh_gnn_rollout 은 torch_geometric 이 필요합니다."
)

from naviertwin.core.cfd_reader.base import CFDDataset  # noqa: E402
from naviertwin.core.gnn.meshgraphnets.rollout_dataset import (  # noqa: E402
    case_to_rollout_trajectory,
)
from naviertwin.web import service  # noqa: E402

# 테스트 속도용 소형 설정 — seed 는 MeshGraphNets 기본값(0)으로 결정적이다
# (n_traj=1 이라 fit() 내부 permutation 셔플도 항상 자명해 순서 비결정성이
# 끼어들 여지가 없다).
_TINY = {"hidden": 8, "n_msgpass": 1, "max_epochs": 25, "device": "cpu"}


def _tiny_unsteady_case(nx: int = 6, ny: int = 6, n_steps: int = 5) -> CFDDataset:
    """단일 케이스, 여러 타임스텝의 초소형 합성 비정상 케이스 (재샘플 스모크용).

    스칼라 ``p`` + 벡터 ``U`` 가 매 스텝 부드럽게 진화한다 — 실제 물리 정확도는
    주장하지 않고, 롤아웃 배관(그래프 빌드/학습/캐싱/외삽)이 유한한 값을
    내는지만 확인한다.
    """
    import pyvista as pv

    image = pv.ImageData(dimensions=(nx, ny, 1), spacing=(1.0, 1.0, 1.0))
    n_points = int(image.n_points)
    pts = np.asarray(image.points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    times = np.linspace(0.0, 1.0, n_steps)

    pressure = np.zeros((n_steps, n_points), dtype=np.float64)
    velocity = np.zeros((n_steps, n_points, 3), dtype=np.float64)
    for i, t in enumerate(times):
        pressure[i] = np.sin(x * 0.5 + t) * np.cos(y * 0.5)
        velocity[i, :, 0] = np.cos(x * 0.5 - t)
        velocity[i, :, 1] = np.sin(y * 0.5 + 0.5 * t)

    image.point_data["p"] = pressure[0]
    image.point_data["U"] = velocity[0]

    return CFDDataset(
        mesh=image,
        time_steps=[float(t) for t in times],
        field_names=["p", "U"],
        metadata={
            "source": "test_mgn_rollout_tiny",
            "time_series_fields": {"p": pressure, "U": velocity},
            "time_series_locations": {"p": "point", "U": "point"},
        },
    )


@pytest.fixture(scope="module")
def unsteady_case() -> CFDDataset:
    return _tiny_unsteady_case()


@pytest.fixture(scope="module")
def rollout_twin(unsteady_case: CFDDataset) -> dict:
    """단일 케이스로 학습한 mesh_gnn_rollout 트윈 (module 공유 — 재학습 방지)."""
    return service.build_mgn_rollout_twin(unsteady_case, ["p", "U"], **_TINY)


# ──────────────────────────────────────────────────────────────────────
# 그래프+트레젝토리 빌더
# ──────────────────────────────────────────────────────────────────────


def test_case_to_rollout_trajectory_shapes(unsteady_case: CFDDataset) -> None:
    built = case_to_rollout_trajectory(unsteady_case, ["p", "U"])
    n_points = int(unsteady_case.n_points)
    n_steps = int(unsteady_case.n_time_steps)

    assert built["target_names"] == ["p", "U_x", "U_y", "U_z"]
    assert built["trajectories"].shape == (1, n_steps, n_points, 4)
    assert built["trajectories"].dtype == np.float32
    assert built["edge_index"].shape[0] == 2
    assert built["edge_features"].shape == (built["edge_index"].shape[1], 4)
    assert built["points"].shape == (n_points, 3)
    assert built["times"] == pytest.approx(list(unsteady_case.time_steps))
    assert np.isfinite(built["trajectories"]).all()


def test_case_to_rollout_trajectory_rejects_too_few_steps() -> None:
    case = _tiny_unsteady_case(n_steps=2)
    with pytest.raises(ValueError, match="타임스텝"):
        case_to_rollout_trajectory(case, ["p"])


def test_case_to_rollout_trajectory_rejects_missing_fields(
    unsteady_case: CFDDataset,
) -> None:
    with pytest.raises(ValueError, match="출력 필드"):
        case_to_rollout_trajectory(unsteady_case, [])


# ──────────────────────────────────────────────────────────────────────
# service.build_mgn_rollout_twin — fit/predict 계약
# ──────────────────────────────────────────────────────────────────────


def test_build_mgn_rollout_twin_contract(
    unsteady_case: CFDDataset, rollout_twin: dict
) -> None:
    engine = rollout_twin["engine"]
    meta = engine.training_metadata

    assert engine.is_fitted
    assert meta["varying_mesh"] is True
    assert meta["problem_type"] == "rollout_forecast"
    assert meta["reducer"] == "mgn_rollout"
    assert meta["surrogate"] == "meshgraphnets_rollout"
    assert meta["param_min"] == pytest.approx(0.0)
    assert meta["param_max"] == pytest.approx(1.0)
    assert meta["n_train_steps"] == unsteady_case.n_time_steps - 1
    # 재샘플 경로 표시(common_grid)가 아니다.
    assert not meta.get("common_grid")

    specs = engine.model.output_fields
    n_points = int(unsteady_case.n_points)
    assert [s["display_name"] for s in specs] == ["p", "U_x", "U_y", "U_z"]
    assert [(s["start"], s["end"]) for s in specs] == [
        (i * n_points, (i + 1) * n_points) for i in range(4)
    ]

    prediction = engine.predict(np.asarray([0.0]))
    assert prediction.shape == (4 * n_points,)
    assert np.isfinite(prediction).all()

    parts = service.split_multi_prediction(engine, prediction)
    assert parts is not None
    assert [name for name, _ in parts] == ["p", "U_x", "U_y", "U_z", "U_mag"]


def test_rollout_predicts_finite_beyond_training_horizon(rollout_twin: dict) -> None:
    """학습 구간 밖(미래) t 도 유한한 값을 낸다 — 진짜 자기회귀 예보의 핵심."""
    engine = rollout_twin["engine"]
    meta = engine.training_metadata
    train_max = float(meta["param_max"])
    future_t = train_max + 3.0 * float(meta["dt"])

    prediction = engine.predict(np.asarray([future_t]))
    assert np.isfinite(prediction).all()

    # 학습 구간 안(내삽)과 다른 값이어야 한다 — 항상 t=0 상태를 그대로
    # 돌려주는 게 아니라 실제로 롤아웃했다는 증거.
    at_zero = engine.predict(np.asarray([0.0]))
    assert not np.allclose(prediction, at_zero)


def test_rollout_caching_avoids_recomputation_from_scratch(
    unsteady_case: CFDDataset,
) -> None:
    """이미 계산한 것보다 더 먼 미래를 요청하면 캐시된 지점에서 이어서 굴린다.

    독립 엔진(모듈 공유 fixture 를 쓰지 않음 — 캐시 상태를 이 테스트가 직접
    통제해야 한다)에서, operator.predict 호출을 감시해 두 번째(더 가까운)
    질의가 첫 질의보다 훨씬 적은 스텝만 굴리는지 확인한다.
    """
    result = service.build_mgn_rollout_twin(unsteady_case, ["p"], **_TINY)
    engine = result["engine"]
    dt = float(engine.training_metadata["dt"])
    t0 = float(engine.training_metadata["param_min"])

    calls: list[int] = []
    original_predict = engine.operator.predict

    def _spy_predict(inputs: dict) -> np.ndarray:
        calls.append(int(inputs["n_steps"]))
        return original_predict(inputs)

    engine.operator.predict = _spy_predict

    far_t = t0 + 10 * dt
    engine.predict(np.asarray([far_t]))
    assert calls == [10]

    # 캐시된 범위 안의 더 가까운 시점 — 새 롤아웃 호출이 전혀 없어야 한다.
    near_t = t0 + 4 * dt
    engine.predict(np.asarray([near_t]))
    assert calls == [10], "캐시 범위 안의 과거 시점은 재계산 없이 바로 반환돼야 한다"

    # 캐시보다 더 먼 시점 — 캐시된 지점(스텝 10)에서 "이어서" 5 스텝만 더 굴려야
    # 한다(처음부터 15 스텝을 다시 굴리지 않음).
    farther_t = t0 + 15 * dt
    engine.predict(np.asarray([farther_t]))
    assert calls == [10, 5]


def test_predict_to_mesh_on_original_case_mesh(
    unsteady_case: CFDDataset, rollout_twin: dict
) -> None:
    """Route 2 존재 증명 — 예측이 원본 케이스 메쉬 위에 재샘플 없이 그대로 붙는다."""
    engine = rollout_twin["engine"]
    predicted, attached = service.predict_to_mesh(engine, [0.5], unsteady_case)
    assert predicted.n_points == unsteady_case.n_points
    assert attached == ["twin_p", "twin_U_x", "twin_U_y", "twin_U_z"]
    for name in attached:
        values = np.asarray(predicted.mesh.point_data[name])
        assert values.shape == (unsteady_case.n_points,)
        assert np.isfinite(values).all()


def test_predict_at_rejects_mismatched_point_count(rollout_twin: dict) -> None:
    engine = rollout_twin["engine"]
    bad_coords = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="단일 케이스"):
        engine.model.predict_at(bad_coords, np.asarray([0.0]))


def test_engine_save_load_bit_identical(rollout_twin: dict, tmp_path) -> None:
    engine = rollout_twin["engine"]
    prediction = engine.predict(np.asarray([0.5]))

    path = tmp_path / "mgn_rollout_engine.pkl"
    engine.save(path)
    from naviertwin.core.digital_twin.mgn_rollout_engine import MGNRolloutTwinEngine

    restored = MGNRolloutTwinEngine.load(path)
    assert np.array_equal(restored.predict(np.asarray([0.5])), prediction)
    assert restored.training_metadata["varying_mesh"] is True


# ──────────────────────────────────────────────────────────────────────
# 전략 레지스트리 — 다른 7개와 정반대 방향 (케이스 세트 거절, 단일 케이스 허용)
# ──────────────────────────────────────────────────────────────────────


def test_strategy_spec_registered() -> None:
    from naviertwin.core.digital_twin.strategies import STRATEGIES

    spec = next(s for s in STRATEGIES if s.key == "mesh_gnn_rollout")
    assert spec.supports_case_sets is False
    assert spec.single_case_needs_steps == 3
    assert spec.tier == "experimental"


def test_strategy_report_accepts_single_case_enough_steps(
    unsteady_case: CFDDataset,
) -> None:
    from naviertwin.core.digital_twin import strategies

    profile = strategies.profile_data(unsteady_case)
    report = strategies.strategy_report(profile)
    assert report["mesh_gnn_rollout"]["ok"] is True


def test_strategy_report_rejects_single_case_too_few_steps() -> None:
    from naviertwin.core.digital_twin import strategies

    case = _tiny_unsteady_case(n_steps=2)
    profile = strategies.profile_data(case)
    report = strategies.strategy_report(profile)
    assert report["mesh_gnn_rollout"]["ok"] is False


def test_strategy_report_rejects_case_sets() -> None:
    from naviertwin.core.digital_twin import strategies

    steady_cases = [_tiny_unsteady_case(n_steps=1) for _ in range(3)]
    profile = strategies.profile_data(steady_cases[0], steady_cases)
    report = strategies.strategy_report(profile)
    assert report["mesh_gnn_rollout"]["ok"] is False
    assert "케이스 세트" in report["mesh_gnn_rollout"]["reason"]
    assert "mesh_gnn_mp" in report["mesh_gnn_rollout"]["reason"]


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


def test_app_mgn_rollout_flow(unsteady_case: CFDDataset) -> None:
    """단일 케이스 시계열 데모 로드 → mesh_gnn_rollout 학습 → 원본 메쉬 위 예측."""
    app = _make_app("nt-test-mgn-rollout")
    st = app.server.state

    app._set_dataset(unsteady_case, status="synthetic-rollout-case")
    assert st.nt_error == ""
    assert st.nt_case_mode is False
    # 능력 레지스트리: 단일 케이스 + 타임스텝 5개 → mesh_gnn_rollout 카드가 켜진다.
    assert st.nt_strategy_status["mesh_gnn_rollout"]["ok"] is True
    assert st.nt_strategy_status["mesh_gnn_rollout"]["tier"] == "experimental"

    st.nt_model_method = "mesh_gnn_rollout"
    st.nt_train_fields = ["p", "U"]
    st.nt_mgn_rollout_epochs = _TINY["max_epochs"]
    st.nt_mgn_rollout_hidden = _TINY["hidden"]
    st.nt_mgn_rollout_msgpass = _TINY["n_msgpass"]
    app.build_twin()
    assert st.nt_error == ""
    assert st.nt_model_ready is True
    assert st.nt_twin_ready is True
    assert "메쉬 GNN 롤아웃" in st.nt_model_summary
    assert app.engine.training_metadata["varying_mesh"] is True

    viewed_points = int(app.dataset.n_points)
    app.state.nt_twin_param = 0.5
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


def test_app_mgn_rollout_case_set_is_rejected() -> None:
    """케이스 세트 + mesh_gnn_rollout 은 단일 케이스 시계열 안내 에러를 낸다."""
    app = _make_app("nt-test-mgn-rollout-case-set")
    st = app.server.state

    datasets = [_tiny_unsteady_case(n_steps=1) for _ in range(3)]
    for dataset in datasets:
        dataset.mesh.point_data["p"] = dataset.mesh.point_data["p"] + 1.0
    result = {
        "datasets": datasets,
        "params": np.asarray([1.0, 2.0, 3.0]).reshape(-1, 1),
        "param_names": ["mu"],
        "case_names": [f"case_{i}" for i in range(len(datasets))],
        "params_source": "synthetic_test",
        "resampled": False,
        "grid_summary": "",
    }
    app._set_case_set(result, "synthetic://mgn-rollout-caseset-test")
    assert st.nt_error == ""
    assert st.nt_case_mode is True
    assert st.nt_strategy_status["mesh_gnn_rollout"]["ok"] is False

    st.nt_model_method = "mesh_gnn_rollout"
    app.build_twin()
    assert "단일 케이스" in st.nt_error
