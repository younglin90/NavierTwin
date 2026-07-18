"""EZyRB 서로게이트(POD-GPR / POD-NN) 배선 테스트 (v5.2).

:mod:`naviertwin.core.surrogate.ezyrb_surrogate` 는 mathLab EZyRB 의
GPR/ANN 근사기를 :class:`BaseSurrogate` 계약(fit/predict)에 맞춰 래핑한다.
이 테스트는 서비스 계층 계약을 검증한다:

- ``service.build_twin(..., surrogate="ezyrb_gpr")`` — 시계열(시간→필드) 경로.
- ``service.build_twin_from_cases(..., surrogate="ezyrb_gpr")`` — 케이스 세트
  (운전조건→필드) 경로.
- POD-GPR 은 학습점을 낮은 상대 L2 오차로 재현해야 하고(보간 성질),
  예측 std 평균이 ``training_metadata["uq_mean_std"]`` 로 노출되어야 한다.
- ``ezyrb_ann`` (POD-NN) 은 작은 설정으로 fit/predict 스모크가 성립해야 한다.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="웹 서비스 테스트에는 pyvista 가 필요합니다.")
pytest.importorskip("ezyrb", reason="EZyRB 서로게이트 테스트에는 ezyrb 가 필요합니다.")

from naviertwin.web import service  # noqa: E402


@pytest.fixture(scope="module")
def demo():
    """작은 시계열 데모 데이터셋 (문제 유형 A)."""
    return service.make_demo_dataset(nx=16, ny=16, n_steps=8)


@pytest.fixture(scope="module")
def sweep():
    """정상 파라미터 스윕 케이스 세트 데모 (문제 유형 B — 5 케이스, μ 2차원)."""
    return service.make_demo_case_set("sweep", n_side=16)


# ──────────────────────────────────────────────────────────────────────
# 등록 — 문자열 키가 서비스/엔진 팩토리에 배선되었는지
# ──────────────────────────────────────────────────────────────────────


def test_ezyrb_keys_registered_in_surrogates() -> None:
    """서비스 SURROGATES 에 ezyrb 키 2개가 등록되고, 느린 ann 은 리더보드
    기본 조합(LEADERBOARD_SURROGATES)에서 빠진다."""
    assert "ezyrb_gpr" in service.SURROGATES
    assert "ezyrb_ann" in service.SURROGATES
    # 기존 키는 그대로 유지된다.
    assert "rbf" in service.SURROGATES
    assert "kriging" in service.SURROGATES
    assert "ezyrb_gpr" in service.LEADERBOARD_SURROGATES
    assert "ezyrb_ann" not in service.LEADERBOARD_SURROGATES


# ──────────────────────────────────────────────────────────────────────
# POD-GPR — 시계열 경로 (build_twin)
# ──────────────────────────────────────────────────────────────────────


def test_build_twin_ezyrb_gpr_predicts_finite_and_time_varying(demo) -> None:
    """(시간→필드) 트윈이 ezyrb_gpr 로 학습되고, 서로 다른 t 에서 유한하며
    서로 다른 필드를 예측한다."""
    result = service.build_twin(demo, "p", n_modes=4, surrogate="ezyrb_gpr")
    assert result["surrogate"] == "ezyrb_gpr"
    engine = result["engine"]

    t0, t1 = result["param_min"], result["param_max"]
    pred_a = service.predict_twin(engine, t0 + 0.25 * (t1 - t0))
    pred_b = service.predict_twin(engine, t0 + 0.75 * (t1 - t0))
    assert pred_a.shape == (demo.n_points,)
    assert pred_b.shape == (demo.n_points,)
    assert np.isfinite(pred_a).all()
    assert np.isfinite(pred_b).all()
    # 감쇠/이류 유동이므로 시간이 다르면 예측 필드도 달라야 한다.
    assert not np.allclose(pred_a, pred_b)


def test_build_twin_ezyrb_gpr_reproduces_training_point(demo) -> None:
    """GPR 은 보간 성질이 있어 학습 시점의 스냅샷을 낮은 상대 L2 로 재현한다."""
    truth = demo.extract_field_snapshots("p")  # (n_features, n_steps)
    times = np.asarray(demo.time_steps, dtype=np.float64)

    result = service.build_twin(demo, "p", n_modes=6, surrogate="ezyrb_gpr")
    mid = len(times) // 2
    pred = service.predict_twin(result["engine"], float(times[mid]))

    ref = truth[:, mid]
    rel_l2 = float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1e-30))
    assert rel_l2 < 0.15, f"학습점 재현 상대 L2 오차가 너무 큽니다: {rel_l2:.4f}"


def test_build_twin_ezyrb_gpr_records_uq_metadata(demo) -> None:
    """POD-GPR 의 예측 std 평균이 트윈 학습 메타데이터(uq_mean_std)에 남는다."""
    result = service.build_twin(demo, "p", n_modes=4, surrogate="ezyrb_gpr")
    engine = result["engine"]

    meta = engine.training_metadata
    assert "uq_mean_std" in meta
    assert np.isfinite(meta["uq_mean_std"])
    assert meta["uq_mean_std"] >= 0.0
    # surrogate 인스턴스에도 마지막 예측 std 가 남는다.
    assert engine.surrogate.last_std_ is not None
    assert np.all(np.asarray(engine.surrogate.last_std_) >= 0.0)


# ──────────────────────────────────────────────────────────────────────
# POD-GPR — 케이스 세트 경로 (build_twin_from_cases)
# ──────────────────────────────────────────────────────────────────────


def test_build_twin_from_cases_ezyrb_gpr(sweep) -> None:
    """정상 스윕 케이스 세트에서 ezyrb_gpr 로 (운전조건→필드) 트윈이 학습되고,
    조건이 다르면 예측 필드도 달라진다."""
    datasets = sweep["datasets"]
    twin = service.build_twin_from_cases(
        datasets,
        "p",
        3,
        sweep["params"],
        param_names=sweep["param_names"],
        surrogate="ezyrb_gpr",
    )
    assert twin["surrogate"] == "ezyrb_gpr"
    assert "uq_mean_std" in twin["engine"].training_metadata

    # 학습 조건 사이(보간)와 다른 조건에서 예측 — 둘 다 유한, 서로 달라야 함.
    lo = np.asarray(twin["param_mins"], dtype=np.float64)
    hi = np.asarray(twin["param_maxs"], dtype=np.float64)
    pred_lo = service.predict_twin(twin["engine"], lo + 0.25 * (hi - lo))
    pred_hi = service.predict_twin(twin["engine"], lo + 0.75 * (hi - lo))
    assert pred_lo.shape == (datasets[0].n_points,)
    assert np.isfinite(pred_lo).all()
    assert np.isfinite(pred_hi).all()
    assert not np.allclose(pred_lo, pred_hi)


# ──────────────────────────────────────────────────────────────────────
# POD-NN (ezyrb_ann) — 스모크
# ──────────────────────────────────────────────────────────────────────


def test_ezyrb_ann_surrogate_class_smoke() -> None:
    """EzyRBANNSurrogate 를 작은 설정(epoch 캡)으로 직접 fit/predict —
    BaseSurrogate 출력 shape 계약(다중 출력 2D, 단일 출력 1D)을 지킨다."""
    from naviertwin.core.surrogate.ezyrb_surrogate import EzyRBANNSurrogate

    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, (20, 2))
    y = np.column_stack([np.sin(X[:, 0]), X[:, 1] ** 2, X[:, 0] * X[:, 1]])

    ann = EzyRBANNSurrogate(layers=(8, 8), max_epochs=200)
    ann.fit(X, y)
    assert ann.is_fitted
    assert ann.input_dim == 2
    assert ann.output_dim == 3

    pred = ann.predict(X)
    assert pred.shape == (20, 3)
    assert np.isfinite(pred).all()
    # 1D 입력(단일 샘플)도 허용 — (1, n_outputs) 로 반환.
    single = ann.predict(X[0])
    assert single.shape == (1, 3)

    # 단일 출력이면 1D 로 평탄화 (rbf/kriging 과 동일 계약).
    ann1 = EzyRBANNSurrogate(layers=(8,), max_epochs=100)
    ann1.fit(X, y[:, 0])
    assert ann1.predict(X).shape == (20,)


def test_build_twin_ezyrb_ann_service_smoke() -> None:
    """서비스 경로에서 ezyrb_ann(POD-NN) 트윈이 학습되고 유한한 시간 가변
    예측을 낸다 (작은 데이터셋 — 속도 우선 스모크)."""
    ds = service.make_demo_dataset(nx=10, ny=10, n_steps=5)
    result = service.build_twin(ds, "p", n_modes=3, surrogate="ezyrb_ann")
    assert result["surrogate"] == "ezyrb_ann"

    t0, t1 = result["param_min"], result["param_max"]
    pred_a = service.predict_twin(result["engine"], t0)
    pred_b = service.predict_twin(result["engine"], t1)
    assert pred_a.shape == (ds.n_points,)
    assert np.isfinite(pred_a).all()
    assert np.isfinite(pred_b).all()
    assert not np.allclose(pred_a, pred_b)
