"""MLflow 실험 추적 테스트 (외부 검토 §6½ #6, 저장 계층 — 실험 관리).

지키려는 계약:
    (a) :meth:`ExperimentTracker.log_run` 으로 기록한 run 이
        :meth:`ExperimentTracker.list_runs` 에 strategy/params/metrics 그대로
        나타난다.
    (b) mlflow 미설치 시뮬레이션(monkeypatch 로 import 실패 강제) →
        ``log_run`` 이 예외 없이 None 을 돌려주고 ``list_runs`` 는 빈 리스트.
    (c) ``build_geometry_fno_twin`` 학습(스모크: epochs≤10, resolution≤16)
        후 ``service.list_training_runs("geometry_fno")`` 에 최소 1건 나타난다.

원칙: 실험 추적은 학습을 절대 막지 않는다 — (b)가 그 계약을 직접 검증한다.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from naviertwin.core.experiment.tracking import ExperimentTracker


@pytest.fixture()
def tracker(tmp_path: Path) -> ExperimentTracker:
    """테스트 전용 tracking 디렉토리를 쓰는 추적기 (실제 ~/.naviertwin 오염 방지)."""
    return ExperimentTracker(tracking_dir=tmp_path / "mlruns", experiment_name="test_exp")


# ──────────────────────────────────────────────────────────────────────
# (a) log_run → list_runs
# ──────────────────────────────────────────────────────────────────────


def test_log_run_appears_in_list_runs(tracker: ExperimentTracker) -> None:
    pytest.importorskip("mlflow", reason="mlflow 가 필요합니다.")
    run_id = tracker.log_run(
        "geometry_fno",
        params={"resolution": 16, "epochs": 5},
        metrics={"remap_floor_rel_l2": 0.05, "loss": 0.9},
        tags={"note": "unit-test"},
    )
    assert run_id is not None

    runs = tracker.list_runs("geometry_fno")
    assert len(runs) == 1
    run = runs[0]
    assert run["run_id"] == run_id
    assert run["strategy"] == "geometry_fno"
    # mlflow 는 param 값을 문자열로 저장한다.
    assert run["params"]["resolution"] == "16"
    assert run["params"]["epochs"] == "5"
    assert run["metrics"]["remap_floor_rel_l2"] == pytest.approx(0.05)
    assert run["metrics"]["loss"] == pytest.approx(0.9)
    assert run["start_time"] is not None


def test_list_runs_filters_by_strategy(tracker: ExperimentTracker) -> None:
    pytest.importorskip("mlflow", reason="mlflow 가 필요합니다.")
    tracker.log_run("geometry_fno", {"resolution": 16}, {"loss": 1.0})
    tracker.log_run("mesh_gnn", {"hidden": 64}, {"loss": 2.0})

    fno_runs = tracker.list_runs("geometry_fno")
    gnn_runs = tracker.list_runs("mesh_gnn")
    all_runs = tracker.list_runs()

    assert len(fno_runs) == 1 and fno_runs[0]["strategy"] == "geometry_fno"
    assert len(gnn_runs) == 1 and gnn_runs[0]["strategy"] == "mesh_gnn"
    assert len(all_runs) == 2


# ──────────────────────────────────────────────────────────────────────
# (b) mlflow 미설치 시뮬레이션 → 조용한 no-op
# ──────────────────────────────────────────────────────────────────────


def test_missing_mlflow_is_silent_noop(
    tracker: ExperimentTracker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mlflow import 자체가 실패해도 log_run/list_runs 가 예외 없이 안전값을 준다."""
    original_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("simulated missing mlflow")
        return original_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    run_id = tracker.log_run("rom", {"n_modes": 4}, {"rmse": 0.1})
    assert run_id is None
    assert tracker.list_runs() == []
    assert tracker.list_runs("rom") == []


# ──────────────────────────────────────────────────────────────────────
# (c) build_geometry_fno_twin 학습 스모크 → service.list_training_runs 반영
# ──────────────────────────────────────────────────────────────────────


def test_geometry_fno_training_run_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """학습 성공 후 앱이 하는 것처럼 log_training_run 을 부르면 조회된다."""
    pytest.importorskip("pyvista", reason="케이스 텐서화에 pyvista 가 필요합니다.")
    pytest.importorskip("torch", reason="GeometryFNO 학습에 torch 가 필요합니다.")
    pytest.importorskip("mlflow", reason="mlflow 가 필요합니다.")

    from naviertwin.web import service

    # 모듈 레벨 lazy-singleton 을 테스트 전용 tracking 디렉토리로 교체 —
    # 실제 ~/.naviertwin/mlruns 를 건드리지 않는다.
    test_tracker = ExperimentTracker(
        tracking_dir=tmp_path / "mlruns", experiment_name="test_exp_geometry_fno"
    )
    monkeypatch.setattr(service, "_experiment_tracker", test_tracker)

    case_set = service.make_demo_case_set("shapes")
    result = service.build_geometry_fno_twin(
        case_set["datasets"],
        "p",
        case_set["params"],
        param_names=case_set["param_names"],
        resolution=16,
        epochs=8,
        modes=4,
        width=8,
    )

    run_id = service.log_training_run(
        "geometry_fno",
        params={
            "resolution": 16,
            "epochs": 8,
            "n_cases": result["n_cases"],
        },
        metrics={"remap_floor_rel_l2": float(result.get("remap_floor_rel_l2", 0.0))},
    )
    assert run_id is not None

    runs = service.list_training_runs("geometry_fno")
    assert len(runs) >= 1
    assert runs[0]["strategy"] == "geometry_fno"
