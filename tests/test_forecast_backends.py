"""LSTM/KNO/LatentODE/NeuralODE 예보 백엔드 스모크 테스트 (계열 Ⓓ, DMD 대안).

``core/time_series/`` 와 ``core/operator_learning/koopman/`` 에 이미 완성돼
있던 예보기 4종을 :func:`naviertwin.web.service.build_forecast_twin` /
:func:`build_parametric_forecast_twin` 로 처음 배선한다. 각 백엔드마다:

  (a) 단일 케이스(시계열) 학습/예측이 finite.
  (b) 케이스 세트(비정상 스윕) 학습/예측이 finite — 학습에 없던 μ 도 포함.

DMD 회귀는 :mod:`tests.test_web_service` 의 기존 테스트가 담당하므로, 여기서는
새 코드가 DMD 경로를 건드리지 않았는지만 별도로 재확인한다(스모크).

epochs/hidden/latent 은 스모크 스케일로 작게 잡아 전체 파일이 3분 이내에
끝나도록 한다.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="웹 서비스 테스트에는 pyvista 가 필요합니다.")
pytest.importorskip("torch", reason="예보 백엔드에는 torch 가 필요합니다.")

from naviertwin.web import service  # noqa: E402

BACKENDS = ("lstm", "koopman_no", "latent_ode", "neural_ode")


def _kwargs_for(backend: str) -> dict:
    """백엔드별 스모크 스케일 하이퍼파라미터 (epochs 소량)."""
    if backend == "lstm":
        return {"max_epochs": 6, "hidden": 8, "lookback": 3}
    if backend == "koopman_no":
        return {"max_epochs": 6, "hidden": 8, "latent": 4}
    if backend == "latent_ode":
        return {"max_epochs": 6, "hidden": 8, "latent": 4, "field_hidden": 8}
    if backend == "neural_ode":
        return {"max_epochs": 6, "hidden": 8}
    raise ValueError(backend)


@pytest.fixture(scope="module")
def single_case():
    """작은 단일 케이스 시계열 — 타임스텝 8개."""
    return service.make_demo_dataset(nx=8, ny=8, n_steps=8)


@pytest.fixture(scope="module")
def case_set():
    """비정상 파라미터 스윕 — 케이스 4개 × 타임스텝 8개, 작은 격자."""
    return service.make_demo_case_set("sweep_unsteady", n_side=8)


# ──────────────────────────────────────────────────────────────────────
# (a) 단일 케이스(시계열)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("backend", BACKENDS)
def test_single_case_forecast_finite(single_case, backend) -> None:
    result = service.build_forecast_twin(
        single_case, "p", backend=backend, n_modes=4, **_kwargs_for(backend)
    )
    engine = result["engine"]
    assert result["backend"] == backend
    assert np.isfinite(result["reconstruction_error"])
    # 존재 이유 — 예보 상한이 학습 구간 상한을 넘는다(DMD 와 같은 계약).
    assert result["forecast_max"] > result["param_max"]

    pred_train = np.asarray(engine.predict([result["param_min"]]))
    assert pred_train.shape[0] > 0
    assert np.isfinite(pred_train).all()

    pred_future = np.asarray(engine.predict([result["forecast_max"]]))
    assert np.isfinite(pred_future).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_single_case_rejects_short_series(backend) -> None:
    ds = service.make_demo_dataset(nx=8, ny=8, n_steps=2)
    with pytest.raises(ValueError, match="4개 이상"):
        service.build_forecast_twin(ds, "p", backend=backend)


# ──────────────────────────────────────────────────────────────────────
# (b) 케이스 세트 (비정상 파라미터 스윕)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("backend", BACKENDS)
def test_case_set_forecast_finite(case_set, backend) -> None:
    result = service.build_parametric_forecast_twin(
        case_set["datasets"],
        "p",
        case_set["params"],
        param_names=case_set["param_names"],
        backend=backend,
        n_modes=4,
        **_kwargs_for(backend),
    )
    engine = result["engine"]
    assert result["backend"] == backend
    assert result["param_names"] == ["inlet_velocity", "t"]
    assert np.isfinite(result["reconstruction_error"])
    assert result["forecast_t_max"] > result["train_t_max"]

    mus = case_set["params"][:, 0]
    known_mu = float(mus[0])
    pred_train = np.asarray(engine.predict([known_mu, result["train_t_max"]]))
    assert np.isfinite(pred_train).all()

    pred_future = np.asarray(engine.predict([known_mu, result["forecast_t_max"]]))
    assert np.isfinite(pred_future).all()

    # 학습에 없던 μ (초기조건 보간 conditioning) 도 유한해야 한다.
    mid_mu = float(0.5 * (mus[0] + mus[1]))
    pred_mid = np.asarray(engine.predict([mid_mu, result["train_t_max"]]))
    assert np.isfinite(pred_mid).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_case_set_rejects_steady_sweep(backend) -> None:
    steady = service.make_demo_case_set("sweep")
    with pytest.raises(ValueError, match="타임스텝 4개 이상"):
        service.build_parametric_forecast_twin(
            steady["datasets"], "p", steady["params"],
            param_names=steady["param_names"], backend=backend,
        )


# ──────────────────────────────────────────────────────────────────────
# 미지원 백엔드 / DMD 회귀 없음
# ──────────────────────────────────────────────────────────────────────


def test_unsupported_backend_rejected() -> None:
    ds = service.make_demo_dataset(nx=8, ny=8, n_steps=8)
    with pytest.raises(ValueError, match="지원하지 않는"):
        service.build_forecast_twin(ds, "p", backend="bogus")
    with pytest.raises(ValueError, match="지원하지 않는"):
        service.build_parametric_forecast_twin(
            [ds, ds], "p", np.array([[1.0], [2.0]]), backend="bogus"
        )


def test_dmd_path_unaffected_by_forecast_backends() -> None:
    """새 예보 백엔드 배선이 기존 DMD 경로(build_dmd_twin)를 회귀시키지 않았는지 재확인.

    전면적인 DMD 회귀 검증은 test_web_service.py 가 담당한다 — 여기서는
    forecast_twin_engine.py/build_forecast_twin 추가가 DMD import·동작에
    부작용을 안 남겼는지만 스모크로 재확인한다.
    """
    ds = service.make_demo_dataset(nx=12, ny=12, n_steps=8)
    result = service.build_dmd_twin(ds, "p")
    assert result["method"] == "dmd"
    assert np.isfinite(result["reconstruction_error"])
    assert result["engine"].training_metadata["problem_type"] == "dynamics_forecast"
