"""실제 vs 트윈 비교 (v5.4) 테스트 — 오차장·요약 지표·외삽 인지.

service 계층(:func:`compute_error_field` / :func:`truth_for_params`)은 순수
numpy 로 검증하고, 앱 계층은 스윕 데모 → ROM 학습 → 예측 흐름에서 상태
(nt_truth_available / nt_extrapolating / twin_error 필드)를 검증한다.
핵심 계약: 진실(ground truth)은 학습 샘플과 **정확히 일치하는** 질의에만
존재한다 — 외삽/보간 지점에서는 오차를 계산하지 않는다 (정직 우선).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="웹 서비스 테스트에는 pyvista 가 필요합니다.")

from naviertwin.web import service  # noqa: E402

# ──────────────────────────────────────────────────────────────────────
# compute_error_field — 손으로 검산 가능한 수치
# ──────────────────────────────────────────────────────────────────────


def test_compute_error_field_exact_math() -> None:
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    prediction = np.array([0.0, 1.0, 2.0, 4.0])
    result = service.compute_error_field(truth, prediction)

    np.testing.assert_allclose(result["abs_error"], [0.0, 0.0, 0.0, 1.0])
    # rmse = sqrt((0+0+0+1)/4) = 0.5
    assert result["rmse"] == pytest.approx(0.5)
    # rel_l2 = ||diff|| / ||truth|| = 1 / sqrt(0+1+4+9)
    assert result["rel_l2"] == pytest.approx(1.0 / np.sqrt(14.0))
    assert result["max_error"] == pytest.approx(1.0)
    # r2 = 1 - SS_res/SS_tot = 1 - 1/5 (mean=1.5, SS_tot=2.25+0.25+0.25+2.25)
    assert result["r2"] == pytest.approx(0.8)


def test_compute_error_field_perfect_prediction() -> None:
    truth = np.array([1.0, -2.0, 3.0])
    result = service.compute_error_field(truth, truth.copy())
    np.testing.assert_allclose(result["abs_error"], 0.0)
    assert result["rmse"] == 0.0
    assert result["rel_l2"] == 0.0
    assert result["max_error"] == 0.0
    assert result["r2"] == pytest.approx(1.0)


def test_compute_error_field_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="크기가 달라"):
        service.compute_error_field(np.zeros(4), np.zeros(5))
    with pytest.raises(ValueError, match="빈 배열"):
        service.compute_error_field(np.zeros(0), np.zeros(0))


# ──────────────────────────────────────────────────────────────────────
# truth_for_params — 학습 샘플 일치 판정 (정상/비정상 스윕, 시계열)
# ──────────────────────────────────────────────────────────────────────


def _dummy_engine(problem_type: str = "") -> SimpleNamespace:
    meta = {"problem_type": problem_type} if problem_type else {}
    return SimpleNamespace(training_metadata=meta)


def test_truth_for_params_steady_sweep_exact_row() -> None:
    """정상 스윕: 파라미터 표의 행과 일치하는 질의 → 그 케이스의 스냅샷."""
    result = service.make_demo_case_set("sweep", n_side=16)
    datasets, params = result["datasets"], result["params"]
    engine = _dummy_engine("steady_sweep")

    # 첫 케이스: inlet_velocity=10, angle_of_attack=0.
    truth = service.truth_for_params(datasets, params, engine, [10.0, 0.0], "p")
    assert truth is not None
    expected = np.asarray(
        datasets[0].extract_field_snapshots("p"), dtype=np.float64
    ).reshape(-1)
    np.testing.assert_allclose(truth, expected)

    # 마지막 케이스도 매칭된다 (행 순서 무관).
    truth_last = service.truth_for_params(datasets, params, engine, [30.0, 8.0], "p")
    assert truth_last is not None
    expected_last = np.asarray(
        datasets[-1].extract_field_snapshots("p"), dtype=np.float64
    ).reshape(-1)
    np.testing.assert_allclose(truth_last, expected_last)


def test_truth_for_params_steady_sweep_perturbed_is_none() -> None:
    """정상 스윕: 살짝 벗어난 질의(보간 지점)는 진실이 없다 → None."""
    result = service.make_demo_case_set("sweep", n_side=16)
    datasets, params = result["datasets"], result["params"]
    engine = _dummy_engine("steady_sweep")

    assert service.truth_for_params(datasets, params, engine, [10.01, 0.0], "p") is None
    assert service.truth_for_params(datasets, params, engine, [10.0, 0.5], "p") is None
    assert service.truth_for_params(datasets, params, engine, [12.5, 1.0], "p") is None
    # 축 개수가 다른 질의도 None (계약 위반을 조용히 보정하지 않는다).
    assert service.truth_for_params(datasets, params, engine, [10.0], "p") is None
    # 없는 field 도 None.
    assert (
        service.truth_for_params(datasets, params, engine, [10.0, 0.0], "no_such")
        is None
    )


def test_truth_for_params_unsteady_sweep_matches_case_and_step() -> None:
    """비정상 스윕: (μ 행 일치) AND (t 가 타임스텝과 일치) 일 때만 스냅샷."""
    result = service.make_demo_case_set("sweep_unsteady", n_side=12)
    datasets, params = result["datasets"], result["params"]
    engine = _dummy_engine("unsteady_sweep")
    times = [float(t) for t in datasets[1].time_steps]

    # μ=15 (두 번째 케이스), t = 정확한 스텝 → 해당 시점 스냅샷.
    truth = service.truth_for_params(datasets, params, engine, [15.0, times[3]], "p")
    assert truth is not None
    matrix = np.asarray(datasets[1].extract_field_snapshots("p"), dtype=np.float64)
    np.testing.assert_allclose(truth, matrix[:, 3])

    # μ 는 정확하지만 t 가 스텝 사이 → None (시간 보간도 외삽처럼 취급).
    mid_t = 0.5 * (times[3] + times[4])
    assert service.truth_for_params(datasets, params, engine, [15.0, mid_t], "p") is None
    # μ 가 학습에 없는 값 → None.
    assert (
        service.truth_for_params(datasets, params, engine, [17.5, times[3]], "p") is None
    )
    # 비정상 스윕에 시간 없는 질의 → None (스냅샷 특정 불가).
    assert service.truth_for_params(datasets, params, engine, [15.0], "p") is None


def test_truth_for_params_single_case_time_series() -> None:
    """단일 케이스 시계열(문제 유형 A): 정확한 t 만 스냅샷을 돌려준다."""
    dataset = service.make_demo_dataset(nx=12, ny=12, n_steps=6)
    engine = _dummy_engine()
    times = [float(t) for t in dataset.time_steps]

    truth = service.truth_for_params(dataset, None, engine, [times[2]], "p")
    assert truth is not None
    matrix = np.asarray(dataset.extract_field_snapshots("p"), dtype=np.float64)
    np.testing.assert_allclose(truth, matrix[:, 2])

    # 스텝 사이 t → None. 리스트로 감싼 호출도 같은 계약.
    mid_t = 0.5 * (times[2] + times[3])
    assert service.truth_for_params(dataset, None, engine, [mid_t], "p") is None
    truth_list = service.truth_for_params([dataset], None, engine, [times[2]], "p")
    assert truth_list is not None
    np.testing.assert_allclose(truth_list, matrix[:, 2])


def test_truth_for_params_none_inputs() -> None:
    engine = _dummy_engine()
    assert service.truth_for_params(None, None, engine, [0.0], "p") is None
    assert service.truth_for_params([], None, engine, [0.0], "p") is None


# ──────────────────────────────────────────────────────────────────────
# 앱 계층 — 스윕 데모 → ROM 학습 → 예측에서의 상태 전이
# ──────────────────────────────────────────────────────────────────────

pytest.importorskip("trame", reason="웹 앱 테스트에는 trame 이 필요합니다.")


def _make_app(name: str):
    """이름이 격리된 trame server 로 앱을 생성한다 (state 누수 방지)."""
    from trame.app import get_server

    from naviertwin.web.app import NavierTwinWebApp

    server = get_server(name, client_type="vue3")
    return NavierTwinWebApp(server=server)


def test_app_predict_at_training_point_attaches_error_field() -> None:
    """학습 샘플과 일치하는 예측 → 오차장(twin_error) + 요약 지표가 붙는다."""
    app = _make_app("nt-test-error-field")
    st = app.server.state

    st.nt_demo_kind = "sweep"
    app.load_demo()
    assert st.nt_case_mode is True

    app.build_twin()  # ROM 파라미터 스윕 (rbf)
    assert st.nt_error == ""
    assert st.nt_model_ready is True

    # 첫 학습 케이스의 정확한 운전조건 (inlet_velocity=10, angle_of_attack=0).
    st.nt_twin_params = [10.0, 0.0]
    app.predict()
    assert st.nt_error == ""
    assert st.nt_predicted is True
    assert st.nt_truth_available is True
    assert st.nt_extrapolating is False
    assert "twin_error" in st.nt_fields
    assert st.nt_error_summary != ""
    assert "RMSE" in st.nt_error_summary

    # 학습 지점의 오차는 작아야 한다 — 평균 절대 오차 < 필드 범위의 10%.
    field = app.engine.training_metadata["field_name"]
    truth = np.asarray(
        app.case_datasets[0].extract_field_snapshots(field), dtype=np.float64
    ).reshape(-1)
    error = np.asarray(app.dataset.mesh.point_data["twin_error"], dtype=np.float64)
    assert error.shape == truth.shape
    field_range = float(truth.max() - truth.min())
    assert field_range > 0.0
    assert float(error.mean()) < 0.1 * field_range


def test_app_predict_off_grid_flags_extrapolation() -> None:
    """학습에 없는 운전조건 → 진실 없음: 오차 계산 없이 외삽 안내만 켠다."""
    app = _make_app("nt-test-error-extrapolation")
    st = app.server.state

    st.nt_demo_kind = "sweep"
    app.load_demo()
    app.build_twin()
    assert st.nt_error == ""

    st.nt_twin_params = [12.5, 1.0]  # 케이스 사이의 보간 지점
    app.predict()
    assert st.nt_error == ""
    assert st.nt_predicted is True
    assert st.nt_truth_available is False
    assert st.nt_extrapolating is True
    assert st.nt_error_summary == ""
    assert "twin_error" not in st.nt_fields


def test_app_stale_error_field_removed_on_next_predict() -> None:
    """학습 지점 예측 뒤 외삽 예측을 하면 이전 오차장이 남지 않는다."""
    app = _make_app("nt-test-error-stale")
    st = app.server.state

    st.nt_demo_kind = "sweep"
    app.load_demo()
    app.build_twin()

    st.nt_twin_params = [10.0, 0.0]
    app.predict()
    assert st.nt_truth_available is True
    assert "twin_error" in st.nt_fields

    st.nt_twin_params = [12.5, 1.0]
    app.predict()
    # 이전 예측의 오차장을 새 예측 옆에 남기면 착시 — 제거돼야 한다.
    assert st.nt_truth_available is False
    assert st.nt_extrapolating is True
    assert "twin_error" not in st.nt_fields
    assert st.nt_error_summary == ""
