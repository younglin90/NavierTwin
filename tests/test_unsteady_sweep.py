"""비정상 케이스 세트 (μ, t) 학습 — v5.0 시간축 보존.

예전에는 ``load_case_set`` 이 케이스마다 마지막 스텝만 남겨서, 비정상×다케이스
데이터가 조용히 정상 스윕으로 붕괴됐다. 여기서 지키는 계약:

1. 로드가 시간축을 보존한다 (PVD 우선 규칙 포함).
2. ROM / Physics AI 가 (μ, t) 를 함께 학습하고, 예측이 μ 와 t 양쪽에 실제로
   반응한다 — 파라미터가 늘었다는 것만으론 부족하고 출력이 변해야 한다.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="케이스 세트 테스트에는 pyvista 가 필요합니다.")

from naviertwin.web import service  # noqa: E402


@pytest.fixture(scope="module")
def sweep_t() -> dict:
    return service.make_demo_case_set("sweep_unsteady")


def test_demo_keeps_time_axis(sweep_t) -> None:
    datasets = sweep_t["datasets"]
    assert len(datasets) == 4
    assert all(d.n_time_steps == 8 for d in datasets)
    # 필드가 μ 와 t 양쪽에 대해 실제로 변한다 — 그래야 학습 대상이 된다.
    snaps0 = datasets[0].extract_field_snapshots("p")
    snaps1 = datasets[1].extract_field_snapshots("p")
    assert snaps0.shape[1] == 8
    assert not np.allclose(snaps0[:, 0], snaps0[:, -1])  # t 에 따라 변함
    assert not np.allclose(snaps0[:, 0], snaps1[:, 0])  # μ 에 따라 변함


def test_expand_case_params_over_time(sweep_t) -> None:
    datasets = sweep_t["datasets"]
    params = sweep_t["params"]
    expanded, names, has_time = service.expand_case_params_over_time(
        datasets, params, sweep_t["param_names"]
    )
    assert has_time is True
    assert names == ["inlet_velocity", "t"]
    assert expanded.shape == (4 * 8, 2)
    # 케이스 0 의 8행은 μ 가 같고 t 만 다르다.
    assert np.allclose(expanded[:8, 0], params[0, 0])
    assert len(np.unique(expanded[:8, 1])) == 8

    # 정상(스텝 1개) 케이스 세트는 그대로 통과한다 — 하위호환.
    steady = service.make_demo_case_set("sweep")
    same, same_names, flag = service.expand_case_params_over_time(
        steady["datasets"], steady["params"], steady["param_names"]
    )
    assert flag is False
    assert same_names == steady["param_names"]
    assert same.shape == steady["params"].shape


def test_rom_learns_mu_and_t(sweep_t) -> None:
    """ROM: (μ, t) → 필드. 예측이 μ 에도 t 에도 반응해야 한다."""
    result = service.build_twin_from_cases(
        sweep_t["datasets"],
        "p",
        6,
        sweep_t["params"],
        param_names=sweep_t["param_names"],
    )
    engine = result["engine"]
    assert result["param_names"] == ["inlet_velocity", "t"]
    assert engine.training_metadata["problem_type"] == "unsteady_sweep"

    base = service.predict_twin(engine, [15.0, 0.2])
    dt = service.predict_twin(engine, [15.0, 1.2])
    dmu = service.predict_twin(engine, [25.0, 0.2])
    assert np.isfinite(base).all()
    assert not np.allclose(base, dt), "t 를 바꿔도 예측이 안 변합니다"
    assert not np.allclose(base, dmu), "μ 를 바꿔도 예측이 안 변합니다"

    # 학습 지점 재현이 대략 맞아야 한다 (rel-L2 < 10%).
    truth = sweep_t["datasets"][1].extract_field_snapshots("p")[:, 1]
    t_val = float(sweep_t["datasets"][1].time_steps[1])
    pred = service.predict_twin(engine, [15.0, t_val])
    rel = np.linalg.norm(pred - truth) / max(np.linalg.norm(truth), 1e-12)
    assert rel < 0.1, f"학습 지점 재현 오차가 너무 큽니다: rel={rel:.3f}"


def test_physics_ai_learns_mu_and_t(sweep_t) -> None:
    """Physics AI: (좌표, μ, t) → 필드. 저epochs 스모크 + 반응성."""
    result = service.build_physics_ai_twin_from_cases(
        sweep_t["datasets"],
        "p",
        sweep_t["params"],
        param_names=sweep_t["param_names"],
        hidden=24,
        max_epochs=40,
        max_train_points=6000,
    )
    engine = result["engine"]
    assert result["param_names"] == ["inlet_velocity", "t"]
    assert engine.training_metadata["problem_type"] == "unsteady_sweep"
    # t 범위가 슬라이더 범위로 나온다 (0 ~ 1.4).
    assert result["param_mins"][1] == pytest.approx(0.0)
    assert result["param_maxs"][1] == pytest.approx(1.4)

    base = np.asarray(engine.predict([15.0, 0.2]))
    dt = np.asarray(engine.predict([15.0, 1.2]))
    assert np.isfinite(base).all()
    assert not np.allclose(base, dt), "t 를 바꿔도 예측이 안 변합니다"


def test_parametric_dmd_interpolates_mu_and_forecasts_t(sweep_t) -> None:
    """ParametricDMD (v5.2): 학습에 없던 μ 보간 + 학습 구간 밖 t 예보.

    데모는 진행파(μ 에 비례하는 전파 속도)라 저랭크 선형 동역학 — DMD 적합 데이터.
    """
    result = service.build_parametric_dmd_twin(
        sweep_t["datasets"],
        "p",
        sweep_t["params"],
        param_names=sweep_t["param_names"],
    )
    engine = result["engine"]
    assert result["param_names"] == ["inlet_velocity", "t"]
    # 적합도: 진행파는 DMD 가 잘 맞아야 한다.
    assert result["reconstruction_error"] < 0.05, (
        f"재구성 오차 {result['reconstruction_error']:.3f} — DMD 적합 실패"
    )
    # 예보 상한이 학습 구간(1.4)을 넘는다.
    assert result["forecast_t_max"] > result["train_t_max"] + 1e-9

    # 학습 지점 재현.
    truth = sweep_t["datasets"][1].extract_field_snapshots("p")[:, 2]
    t2 = float(sweep_t["datasets"][1].time_steps[2])
    pred = np.asarray(engine.predict([15.0, t2]))
    rel = np.linalg.norm(pred - truth) / max(np.linalg.norm(truth), 1e-12)
    assert rel < 0.05

    # 학습에 없던 μ=17.5: 이웃(15, 20)의 사이 값이어야 하고 유한해야 한다.
    mid = np.asarray(engine.predict([17.5, t2]))
    assert np.isfinite(mid).all()
    lo = np.asarray(engine.predict([15.0, t2]))
    hi = np.asarray(engine.predict([20.0, t2]))
    # 보간이 이웃보다 멀리 튀지 않는다 (전 지점 부등식은 과도 — 평균 거리로).
    d_mid = min(np.linalg.norm(mid - lo), np.linalg.norm(mid - hi))
    d_far = np.linalg.norm(hi - lo)
    assert d_mid < d_far, "μ 보간이 이웃 케이스 사이에 있지 않습니다"

    # 학습 구간 밖 t 예보가 유한하고, t 에 따라 계속 변한다.
    f1 = np.asarray(engine.predict([15.0, 1.6]))
    f2 = np.asarray(engine.predict([15.0, 2.0]))
    assert np.isfinite(f1).all() and np.isfinite(f2).all()
    assert not np.allclose(f1, f2), "예보 구간에서 시간 전개가 멈췄습니다"


def test_parametric_dmd_rejects_steady_sweep() -> None:
    steady = service.make_demo_case_set("sweep")
    with pytest.raises(ValueError, match="타임스텝 4개 이상"):
        service.build_parametric_dmd_twin(
            steady["datasets"], "p", steady["params"],
            param_names=steady["param_names"],
        )


def test_load_case_set_prefers_pvd_and_keeps_time(tmp_path) -> None:
    """폴더에 pvd 가 있으면 pvd 만 케이스 — 참조된 vtk 가 케이스로 새면 안 된다."""
    import pyvista as pv

    for case_idx, amp in enumerate((1.0, 2.0)):
        refs = []
        for step in range(3):
            grid = pv.ImageData(dimensions=(5, 5, 1))
            grid.point_data["p"] = amp * (step + 1.0) * np.linspace(0, 1, grid.n_points)
            name = f"case{case_idx}_{step:02d}.vtk"
            grid.save(str(tmp_path / name))
            refs.append((name, float(step) * 0.5))
        entries = "\n".join(
            f'    <DataSet timestep="{t}" file="{name}"/>' for name, t in refs
        )
        (tmp_path / f"case{case_idx}.pvd").write_text(
            '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n'
            f"  <Collection>\n{entries}\n  </Collection>\n</VTKFile>\n",
            encoding="utf-8",
        )
    (tmp_path / "params.csv").write_text("amp\n1.0\n2.0\n", encoding="utf-8")

    result = service.load_case_set(tmp_path)
    # vtk 6개가 아니라 pvd 2개만 케이스로 잡힌다.
    assert len(result["datasets"]) == 2
    assert all(name.endswith(".pvd") for name in result["case_names"])
    # 시간축이 보존된다.
    assert all(d.n_time_steps == 3 for d in result["datasets"])
    snaps = result["datasets"][0].extract_field_snapshots("p")
    assert snaps.shape[1] == 3
    assert not np.allclose(snaps[:, 0], snaps[:, -1])


def test_unsteady_varying_mesh_resample_is_refused(tmp_path) -> None:
    """비정상 + 형상 가변 + 재샘플 = 시간을 조용히 버리므로 명시 거절."""
    import pyvista as pv

    from naviertwin.core.cfd_reader.base import CFDDataset  # noqa: F401

    for i, side in enumerate((5, 6)):
        refs = []
        for step in range(2):
            grid = pv.ImageData(dimensions=(side, side, 1))
            grid.point_data["p"] = np.full(grid.n_points, float(step))
            name = f"c{i}_{step}.vtk"
            grid.save(str(tmp_path / name))
            refs.append((name, float(step)))
        entries = "\n".join(
            f'    <DataSet timestep="{t}" file="{n}"/>' for n, t in refs
        )
        (tmp_path / f"c{i}.pvd").write_text(
            '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n'
            f"  <Collection>\n{entries}\n  </Collection>\n</VTKFile>\n",
            encoding="utf-8",
        )
    (tmp_path / "params.csv").write_text("mu\n1.0\n2.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="비정상.*재샘플"):
        service.load_case_set(tmp_path)  # resample="auto" + 메쉬 다름 → 거절
    # resample=False 면 시간축을 지킨 채 로드된다 (Physics AI 용).
    result = service.load_case_set(tmp_path, resample=False)
    assert all(d.n_time_steps == 2 for d in result["datasets"])
