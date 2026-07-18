"""GeometryFNO(FNO+SDF) 웹 배선 테스트 — 케이스 세트 "operator" 전략 (v5.2).

지키려는 계약:
    - ``service.build_geometry_fno_twin`` 이 정상 케이스 세트를 공통 격자
      텐서로 학습하고, 예측이 **공통 격자 점 순서**의 field-major 벡터다.
    - 비정상(시계열) 케이스 세트는 명확한 한국어 메시지로 거절된다.
    - 앱에서 케이스 세트 + method="operator" 로 학습→예측이 끝까지 돌고,
      예측 결과가 공통 격자 뷰(``twin_*`` 필드)로 표시된다.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="케이스 텐서화에 pyvista 가 필요합니다.")
pytest.importorskip("torch", reason="GeometryFNO 학습에 torch 가 필요합니다.")

from naviertwin.web import service  # noqa: E402

# 테스트 속도용 소형 설정 — seed 는 GeometryFNO2D 기본값(0)으로 결정적이다.
# epochs 는 "학습 μ 예측이 자기 케이스를 따라간다" 약한 검증이 통과할 만큼만
# (40 epoch 에서는 아직 평균장 수준이라 케이스 구분이 안 된다 — 결정적 seed
# 라 한 번 통과하면 재현된다).
_TINY = {"modes": 8, "width": 16, "epochs": 160}
_RESOLUTION = 24


@pytest.fixture(scope="module")
def shapes_case_set() -> dict:
    """반지름이 다른 원기둥 5케이스 (정상, 공통 격자로 재샘플된 데모)."""
    return service.make_demo_case_set("shapes")


@pytest.fixture(scope="module")
def shapes_twin(shapes_case_set: dict) -> dict:
    """shapes 데모로 학습한 GeometryFNO 트윈 (module 공유 — 재학습 방지)."""
    return service.build_geometry_fno_twin(
        shapes_case_set["datasets"],
        "p",
        shapes_case_set["params"],
        param_names=shapes_case_set["param_names"],
        resolution=_RESOLUTION,
        **_TINY,
    )


# ──────────────────────────────────────────────────────────────────────
# 서비스 계층
# ──────────────────────────────────────────────────────────────────────


def test_build_returns_engine_with_contract_metadata(shapes_twin: dict) -> None:
    engine = shapes_twin["engine"]
    meta = engine.training_metadata
    assert meta["problem_type"] == "steady_sweep_operator"
    assert meta["reducer"] == "geometry_fno"
    assert meta["surrogate"] == "fno_sdf(builtin)"
    assert meta["common_grid"] is True
    assert meta["n_cases"] == 5
    assert meta["param_names"] == ["radius"]
    # 파라미터 범위가 데모 반지름 [0.06, 0.18] 과 일치한다.
    assert meta["param_mins"] == pytest.approx([0.06])
    assert meta["param_maxs"] == pytest.approx([0.18])
    assert shapes_twin["param_mins"] == pytest.approx([0.06])
    assert shapes_twin["param_maxs"] == pytest.approx([0.18])
    assert shapes_twin["n_cases"] == 5


def test_grid_dataset_ready_for_viewer(shapes_twin: dict) -> None:
    """예측 표시용 grid_dataset: 공통 격자 + sdf/mask 필드."""
    grid_dataset = shapes_twin["engine"].grid_dataset
    assert "sdf" in grid_dataset.mesh.point_data
    assert "mask" in grid_dataset.mesh.point_data
    assert grid_dataset.n_points > 0


def test_predict_is_flat_vector_on_common_grid(shapes_twin: dict) -> None:
    engine = shapes_twin["engine"]
    prediction = engine.predict(np.asarray([0.12]))
    # 단일 출력(p) → 길이 = 공통 격자 점 수 (field-major, 채널 1개).
    assert prediction.shape == (engine.grid_dataset.n_points,)
    assert np.isfinite(prediction).all()


def test_predict_at_training_mu_tracks_its_own_case(
    shapes_case_set: dict, shapes_twin: dict
) -> None:
    """학습 μ 예측이 그 케이스 참값에 (가장 먼 케이스보다) 가깝다 — 약한 검증.

    소표본(5케이스) 학습이라 정량 정확도는 주장하지 않는다 — 모델이 μ/SDF
    채널에 실제로 반응한다는 방향성만 확인한다.
    """
    from naviertwin.core.operator_learning.fno.case_tensorizer import (
        cases_to_grid_tensors,
    )

    # 엔진과 같은 규칙(합집합 바운딩 박스 + 해상도)으로 참값 텐서를 재구성한다.
    tensors = cases_to_grid_tensors(
        shapes_case_set["datasets"],
        shapes_case_set["params"],
        field_names=["p"],
        resolution=_RESOLUTION,
        param_names=shapes_case_set["param_names"],
    )
    truth_first = tensors["targets"][0][:, :, 0].ravel()
    truth_last = tensors["targets"][-1][:, :, 0].ravel()

    engine = shapes_twin["engine"]
    prediction = engine.predict(np.asarray([0.06]))  # 첫 케이스의 학습 μ
    err_own = float(np.sqrt(np.mean((prediction - truth_first) ** 2)))
    err_far = float(np.sqrt(np.mean((prediction - truth_last) ** 2)))
    assert err_own < err_far, (
        f"학습 μ 예측이 자기 케이스보다 가장 먼 케이스에 더 가깝습니다 "
        f"(own={err_own:.4g}, far={err_far:.4g})"
    )


def test_predict_at_unseen_mu_is_finite(shapes_twin: dict) -> None:
    """학습에 없던 μ (형상 채널은 최근접 케이스 재사용) → 유한한 예측."""
    engine = shapes_twin["engine"]
    prediction = engine.predict(np.asarray([0.10]))  # 0.09 와 0.12 사이
    assert np.isfinite(prediction).all()
    # 최근접 케이스 선택이 정규화 거리 기준으로 동작한다.
    assert engine.nearest_case_index(np.asarray([0.061])) == 0
    assert engine.nearest_case_index(np.asarray([0.179])) == 4


def test_predict_rejects_wrong_param_dim(shapes_twin: dict) -> None:
    with pytest.raises(ValueError, match="파라미터 차원"):
        shapes_twin["engine"].predict(np.asarray([0.1, 0.2]))


def test_unsteady_case_set_is_rejected() -> None:
    """비정상 스윕은 명확히 거절한다 — 시계열을 조용히 뭉개지 않는다."""
    result = service.make_demo_case_set("sweep_unsteady")
    with pytest.raises(ValueError, match="미지원"):
        service.build_geometry_fno_twin(
            result["datasets"],
            "p",
            result["params"],
            param_names=result["param_names"],
            resolution=16,
            **_TINY,
        )


def test_multi_field_output_specs_and_split(shapes_case_set: dict) -> None:
    """다중 출력(p + U 성분): output_fields 경계와 split_multi_prediction 계약."""
    built = service.build_geometry_fno_twin(
        shapes_case_set["datasets"],
        ["p", "U"],
        shapes_case_set["params"],
        param_names=shapes_case_set["param_names"],
        resolution=16,
        modes=6,
        width=8,
        epochs=3,  # 배선 검증용 — 정확도는 보지 않는다
    )
    engine = built["engine"]
    specs = engine.model.output_fields
    assert [s["display_name"] for s in specs] == ["p", "U_x", "U_y", "U_z"]
    # 채널명 U_x → 요청 필드 U 로 되돌려 매핑돼야 크기(magnitude) 파생이 된다.
    assert [s["field_name"] for s in specs] == ["p", "U", "U", "U"]

    prediction = engine.predict(np.asarray([0.12]))
    n_grid = engine.grid_dataset.n_points
    assert prediction.shape == (4 * n_grid,)
    parts = service.split_multi_prediction(engine, prediction)
    assert parts is not None
    names = [name for name, _ in parts]
    assert names == ["p", "U_x", "U_y", "U_z", "U_mag"]
    assert all(segment.shape == (n_grid,) for _, segment in parts)


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


def test_app_operator_case_set_flow() -> None:
    """shapes 데모 → operator 전략 가능 판정 → 학습 → 공통 격자 예측."""
    app = _make_app("nt-test-geometry-fno")
    st = app.server.state

    st.nt_demo_kind = "shapes"
    app.load_demo()
    assert st.nt_error == ""
    assert st.nt_case_mode is True
    # 능력 레지스트리: 정상 케이스 세트 5개 → operator 카드가 켜진다.
    assert st.nt_strategy_status["operator"]["ok"] is True

    st.nt_model_method = "operator"
    st.nt_operator_epochs = 30  # 테스트 속도용
    st.nt_operator_resolution = 20
    app.build_twin()
    assert st.nt_error == ""
    assert st.nt_model_ready is True
    assert st.nt_twin_ready is True
    assert "GeometryFNO" in st.nt_model_summary
    # 상태 메시지에 케이스 수/해상도가 담긴다.
    assert "케이스 5개" in st.nt_status
    assert "해상도 20" in st.nt_status
    # 슬라이더 범위 = 학습 반지름 범위, 기본값 = 중앙.
    assert st.nt_param_names == ["radius"]
    assert st.nt_twin_mins == pytest.approx([0.06])
    assert st.nt_twin_maxs == pytest.approx([0.18])

    # 중앙 파라미터(기본값)에서 예측 → 공통 격자 뷰로 교체 + twin_ 필드 표시.
    app.predict()
    assert st.nt_error == ""
    assert "공통 격자" in st.nt_status
    assert "twin_prediction" in st.nt_fields
    assert "sdf" in st.nt_fields  # 공통 격자 데이터셋의 형상 필드도 함께 보인다
    prediction = np.asarray(app.dataset.mesh.point_data["twin_prediction"])
    assert prediction.shape == (app.engine.grid_dataset.n_points,)
    assert np.isfinite(prediction).all()
    # 학습 상태는 보존된다 — 케이스 뷰로 되돌아가도 엔진이 살아 있다.
    assert app.engine is not None
    st.nt_case_index = 2
    app.select_case()
    assert st.nt_error == ""
    assert app.engine is not None


def test_app_single_case_operator_keeps_lab_error() -> None:
    """단일 케이스 + operator 는 여전히 ⑥연산자 랩 안내 에러다."""
    app = _make_app("nt-test-geometry-fno-single")
    st = app.server.state
    app.load_demo()  # filament 시계열 (케이스 세트 아님)
    assert st.nt_case_mode is False
    st.nt_model_method = "operator"
    app.build_twin()
    assert "연산자 랩" in st.nt_error
