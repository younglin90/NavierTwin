"""웹 워크플로우 서비스/렌더 계층 테스트 (Qt/GL 비의존).

:mod:`naviertwin.web.service` 와 :mod:`naviertwin.web.render` 는 trame UI 없이
``core`` 모듈 위에서 Import → Analyze → Reduce → Twin MVP 워크플로우를 조립한다.
이 테스트는 PyVista(필터, GL 불필요)만 사용해 headless 환경에서 동작한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="웹 서비스/렌더 테스트에는 pyvista 가 필요합니다.")

from naviertwin.web import render, service  # noqa: E402


@pytest.fixture(scope="module")
def demo():
    """작은 Taylor–Green 데모 데이터셋."""
    return service.make_demo_dataset(nx=20, ny=20, n_steps=8)


# ──────────────────────────────────────────────────────────────────────
# Import / demo dataset
# ──────────────────────────────────────────────────────────────────────


def test_make_demo_dataset_shapes(demo) -> None:
    info = service.dataset_info(demo)
    assert info["points"] == 20 * 20
    assert info["time_steps"] == 8
    assert info["fields"] == ["U", "p"]
    assert info["source"] == "demo_taylor_green"
    # 시계열 필드가 metadata 에 timestep 축으로 저장된다.
    series = demo.metadata["time_series_fields"]
    assert series["U"].shape == (8, 400, 3)
    assert series["p"].shape == (8, 400)


def test_make_demo_dataset_filament_is_discontinuous_and_evolving() -> None:
    ds = service.make_demo_dataset(nx=40, ny=40, n_steps=8, kind="filament")
    assert service.dataset_info(ds)["source"] == "demo_swirl_filament"
    p = np.asarray(ds.metadata["time_series_fields"]["p"]).reshape(8, 40, 40)
    # 불연속: 인접 셀 점프가 필드 범위에 육박(부드러운 데이터면 훨씬 작음).
    field_range = float(p.max() - p.min())
    max_jump = float(np.abs(np.diff(p[0], axis=1)).max())
    assert max_jump > 0.5 * field_range
    # 시간 진화: 첫/중간 스냅샷이 확연히 다름 (p).
    assert not np.allclose(p[0], p[4])
    # 비정상 유동: 속도장 U 도 시간에 따라 변한다(가속 소용돌이).
    u = np.asarray(ds.metadata["time_series_fields"]["U"])
    assert not np.allclose(u[0], u[4])
    assert not np.allclose(u[4], u[7])


def test_make_demo_dataset_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        service.make_demo_dataset(kind="bogus")


def test_sync_timestep_to_mesh_updates_base_fields(demo) -> None:
    service.sync_timestep_to_mesh(demo, 0)
    u0 = np.asarray(demo.mesh.point_data["U"]).copy()
    service.sync_timestep_to_mesh(demo, 5)
    u5 = np.asarray(demo.mesh.point_data["U"])
    # 감쇠 와류이므로 timestep 이 다르면 속도장도 달라진다.
    assert not np.allclose(u0, u5)


# ──────────────────────────────────────────────────────────────────────
# Analyze — 와류 식별
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method,expected", [("q_criterion", "Q-criterion"), ("lambda2", "lambda2")])
def test_run_vortex_analysis_attaches_field(method, expected) -> None:
    ds = service.make_demo_dataset(nx=16, ny=16, n_steps=4)
    field = service.run_vortex_analysis(ds, method, timestep=1)
    assert field == expected
    assert field in ds.field_names
    assert field in ds.mesh.point_data
    values = np.asarray(ds.mesh.point_data[field])
    assert values.shape[0] == ds.n_points
    assert np.isfinite(values).all()


def test_run_vortex_analysis_rejects_unknown_method(demo) -> None:
    with pytest.raises(ValueError):
        service.run_vortex_analysis(demo, "not_a_method")


# ──────────────────────────────────────────────────────────────────────
# Analyze — FFT / PSD
# ──────────────────────────────────────────────────────────────────────


def test_estimate_dt(demo) -> None:
    dt = service.estimate_dt(demo)
    assert dt > 0
    # 단일 타임스텝이면 1.0 폴백.
    single = service.make_demo_dataset(nx=8, ny=8, n_steps=1)
    assert service.estimate_dt(single) == 1.0


def test_compute_fft_psd_shapes_and_peak() -> None:
    # 충분한 타임스텝으로 주입 진동(1.5Hz)을 검출.
    ds = service.make_demo_dataset(nx=12, ny=12, n_steps=48, oscillation_hz=1.5)
    result = service.compute_fft_psd(ds, "p")
    assert len(result["freqs"]) == len(result["amplitudes"])
    assert len(result["psd_freqs"]) == len(result["psd"])
    assert result["probe"] == "spatial_mean"
    assert result["dt"] > 0
    # 지배 주파수가 주입 진동 근처(±0.6Hz)에 있어야 한다.
    assert result["dominant"], "지배 주파수를 찾지 못했습니다."
    top = result["dominant"][0]["frequency"]
    assert abs(top - 1.5) < 0.6


def test_compute_fft_psd_point_probe_and_dt_override() -> None:
    ds = service.make_demo_dataset(nx=10, ny=10, n_steps=24)
    result = service.compute_fft_psd(ds, "U", point_index=5, dt=0.05)
    assert result["probe"] == "point[5]"
    assert result["dt"] == pytest.approx(0.05)


def test_compute_fft_psd_requires_timeseries() -> None:
    single = service.make_demo_dataset(nx=8, ny=8, n_steps=1)
    with pytest.raises(ValueError):
        service.compute_fft_psd(single, "U")


# ──────────────────────────────────────────────────────────────────────
# Reduce — POD
# ──────────────────────────────────────────────────────────────────────


def test_run_pod_energy_is_monotonic_and_bounded(demo) -> None:
    result = service.run_pod(demo, "U", n_modes=5)
    assert result["n_modes"] == 5
    assert result["n_snapshots"] == 8
    energy = np.asarray(result["cumulative_energy"])
    assert len(result["singular_values"]) == 5
    # 누적 에너지는 단조 증가하고 1.0 을 넘지 않는다.
    assert np.all(np.diff(energy) >= -1e-9)
    assert energy[-1] <= 1.0 + 1e-9
    assert hasattr(result["reducer"], "modes_")


def test_run_pod_clamps_modes_to_snapshot_count(demo) -> None:
    # 요청 모드 수가 스냅샷 수보다 크면 자동으로 클램프된다.
    result = service.run_pod(demo, "p", n_modes=999)
    assert result["n_modes"] <= result["n_snapshots"]


def test_pod_mode_field_extracts_mode(demo) -> None:
    result = service.run_pod(demo, "U", n_modes=3)
    mode0 = service.pod_mode_field(result["reducer"], 0)
    assert mode0.ndim == 1
    assert mode0.shape[0] == demo.n_points


# ──────────────────────────────────────────────────────────────────────
# Twin — 시간 → 필드 예측
# ──────────────────────────────────────────────────────────────────────


def test_build_and_predict_twin(demo) -> None:
    result = service.build_twin(demo, "U", n_modes=5)
    assert result["param_min"] == pytest.approx(0.0)
    assert result["param_max"] > result["param_min"]
    engine = result["engine"]

    mid = 0.5 * (result["param_min"] + result["param_max"])
    prediction = service.predict_twin(engine, mid)
    assert prediction.ndim == 1
    assert prediction.shape[0] == demo.n_points
    assert np.isfinite(prediction).all()

    field = service.attach_prediction(demo, prediction)
    assert field in demo.field_names
    assert field in demo.mesh.point_data


def test_build_twin_requires_multiple_timesteps() -> None:
    single = service.make_demo_dataset(nx=12, ny=12, n_steps=1)
    with pytest.raises(ValueError):
        service.build_twin(single, "U", n_modes=2)


def test_build_twin_metadata_records_param_range(demo) -> None:
    """복원 시 예측 슬라이더가 학습 범위를 벗어나지 않도록 metadata 에 범위 기록."""
    result = service.build_twin(demo, "U", 3)
    meta = result["engine"].training_metadata
    assert meta["param_min"] == pytest.approx(result["param_min"])
    assert meta["param_max"] == pytest.approx(result["param_max"])
    assert meta["param_max"] > meta["param_min"]


@pytest.mark.parametrize("reducer", service.REDUCERS)
@pytest.mark.parametrize("surrogate", service.SURROGATES)
def test_build_twin_reducer_surrogate_combos(reducer, surrogate) -> None:
    ds = service.make_demo_dataset(nx=12, ny=12, n_steps=6)
    result = service.build_twin(ds, "U", 4, reducer=reducer, surrogate=surrogate)
    assert result["reducer"] == reducer
    assert result["surrogate"] == surrogate
    pred = service.predict_twin(result["engine"], 0.5)
    assert pred.shape[0] == ds.n_points
    assert np.isfinite(pred).all()


def test_build_twin_rejects_unknown_reducer(demo) -> None:
    with pytest.raises(ValueError):
        service.build_twin(demo, "U", 3, reducer="bogus")


def test_attach_pod_mode(demo) -> None:
    result = service.run_pod(demo, "U", n_modes=4)
    field = service.attach_pod_mode(demo, result["reducer"], 2)
    assert field == "pod_mode_2"
    assert field in demo.field_names
    assert field in demo.mesh.point_data


# ──────────────────────────────────────────────────────────────────────
# Compare — reducer×surrogate 벤치마크
# ──────────────────────────────────────────────────────────────────────


def test_compare_models_ranks_all_combos() -> None:
    """ROM 조합 역학만 검증 — Physics AI 행은 전용 테스트에서 다룬다."""
    ds = service.make_demo_dataset(nx=12, ny=12, n_steps=8)
    result = service.compare_models(ds, "U", 4, include_physics=False)
    rows = result["rows"]
    assert len(rows) == len(service.REDUCERS) * len(service.SURROGATES)
    # RMSE 오름차순 정렬.
    rmses = [r["rmse"] for r in rows]
    assert rmses == sorted(rmses)
    # 모든 조합이 정상 학습되어야 한다.
    assert all(r["status"] == "ok" for r in rows)
    # best 는 최저 RMSE 행.
    assert result["best"]["rmse"] == min(rmses)
    for row in rows:
        assert {"combo", "reducer", "surrogate", "n_modes", "rmse", "r2", "rel_l2", "latency_ms"} <= set(row)
        assert row["latency_ms"] >= 0


def test_compare_models_subset_combos(demo) -> None:
    result = service.compare_models(
        demo, "U", 3, combos=[("pod", "rbf")], include_physics=False
    )
    assert len(result["rows"]) == 1
    assert result["rows"][0]["combo"] == "pod+rbf"


def test_compare_models_progress_cb(demo) -> None:
    """progress_cb 가 조합 시작마다 + 완료 시 1회 호출된다 (done, total, label)."""
    calls: list[tuple[int, int, str]] = []
    combos = [("pod", "rbf"), ("pod", "kriging")]
    service.compare_models(
        demo,
        "U",
        3,
        combos=combos,
        include_physics=False,
        progress_cb=lambda d, n, lbl: calls.append((d, n, lbl)),
    )
    # 조합 2개 시작(done=0,1) + 완료(done=2) = 3회, total 은 항상 2.
    assert [c[0] for c in calls] == [0, 1, 2]
    assert all(c[1] == 2 for c in calls)


def test_compare_models_requires_timeseries() -> None:
    single = service.make_demo_dataset(nx=8, ny=8, n_steps=1)
    with pytest.raises(ValueError):
        service.compare_models(single, "U", 2)


# ──────────────────────────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────────────────────────


def test_snapshot_dataset_materializes_single_step(demo) -> None:
    snap = service.snapshot_dataset(demo, 3)
    assert snap.n_time_steps == 1
    assert snap.n_points == demo.n_points
    # 시계열 메타데이터는 제거되고 현재 step field 가 메쉬에 직접 부착된다.
    assert "time_series_fields" not in snap.metadata
    assert "U" in snap.mesh.point_data


def test_export_csv_and_vtk(tmp_path, demo) -> None:
    snap = service.snapshot_dataset(demo, 0)
    csv = service.export_field_csv(snap, tmp_path / "fields.csv")
    assert csv.endswith(".csv")
    data = np.loadtxt(csv, delimiter=",", skiprows=1)
    assert data.shape[0] == demo.n_points

    # ImageData 는 .vtu 미지원 → .vtk 로 폴백.
    vtk = service.export_vtk(snap, tmp_path / "mesh.vtu")
    assert vtk.endswith(".vtk")


def test_save_project_roundtrip(tmp_path, demo) -> None:
    from naviertwin.core.export.ntwin_format import load_dataset

    service.run_vortex_analysis(demo, "q_criterion", timestep=1)
    engine = service.build_twin(demo, "U", 4)["engine"]
    snap = service.snapshot_dataset(demo, 1)
    paths = service.save_project(snap, tmp_path / "proj.ntwin", engine=engine)
    assert paths["project"].endswith(".ntwin")
    assert "engine" in paths
    reloaded = load_dataset(__import__("pathlib").Path(paths["project"]))
    assert reloaded.n_points == demo.n_points
    assert reloaded.n_time_steps == 1


def test_save_engine_and_report(tmp_path, demo) -> None:
    engine = service.build_twin(demo, "U", 3)["engine"]
    eng_path = service.save_engine(engine, tmp_path / "eng.pkl")
    assert eng_path.endswith(".pkl")

    report = service.export_report(
        service.snapshot_dataset(demo, 0),
        tmp_path / "report.html",
        model_info={"reducer": "pod"},
        metrics={"energy": 0.999},
    )
    assert report.endswith(".html")
    text = __import__("pathlib").Path(report).read_text(encoding="utf-8")
    assert "NavierTwin" in text


def test_save_engine_requires_engine() -> None:
    with pytest.raises(ValueError):
        service.save_engine(None, "x.pkl")


# ──────────────────────────────────────────────────────────────────────
# Import — .ntwin 프로젝트 열기 / 실제 CFD 파일 로드
# ──────────────────────────────────────────────────────────────────────


def test_load_project_roundtrip(tmp_path, demo) -> None:
    service.run_vortex_analysis(demo, "q_criterion", timestep=1)
    engine = service.build_twin(demo, "U", 4)["engine"]
    snap = service.snapshot_dataset(demo, 1)
    paths = service.save_project(snap, tmp_path / "proj.ntwin", engine=engine)

    dataset, restored = service.load_project(paths["project"])
    assert dataset.n_points == demo.n_points
    assert dataset.n_time_steps == 1
    assert restored is not None
    pred = service.predict_twin(restored, 0.5)
    assert pred.shape[0] == demo.n_points


def test_load_project_without_engine(tmp_path, demo) -> None:
    paths = service.save_project(service.snapshot_dataset(demo, 0), tmp_path / "p2.ntwin")
    _, engine = service.load_project(paths["project"])
    assert engine is None


def test_load_project_rejects_non_ntwin(tmp_path) -> None:
    # 디렉토리(또는 비-.ntwin)는 거부.
    with pytest.raises(ValueError):
        service.load_project(tmp_path)


def test_load_project_missing_path() -> None:
    with pytest.raises(FileNotFoundError):
        service.load_project("/no/such/proj.ntwin")


def test_load_dataset_reads_real_vtk(tmp_path) -> None:
    import pyvista as pv

    mesh = pv.Sphere().cast_to_unstructured_grid()
    mesh.point_data["p"] = np.arange(mesh.n_points, dtype=np.float64)
    out = tmp_path / "case.vtk"
    mesh.save(str(out))

    dataset = service.load_dataset(out)
    assert dataset.n_points == mesh.n_points
    assert "p" in render.available_fields(dataset)


def test_is_derived_field_ssot() -> None:
    assert service.is_derived_field("Q-criterion")
    assert service.is_derived_field("lambda2")
    assert service.is_derived_field("vorticity")
    assert service.is_derived_field("pod_mode_3")
    assert service.is_derived_field("twin_prediction")
    assert not service.is_derived_field("U")
    assert not service.is_derived_field("p")


# ──────────────────────────────────────────────────────────────────────
# Render — Qt 비의존 메쉬 준비
# ──────────────────────────────────────────────────────────────────────


def test_available_and_preferred_fields(demo) -> None:
    names = render.available_fields(demo)
    assert "U" in names and "p" in names
    # p 가 우선순위에서 U 보다 앞선다.
    assert render.preferred_field(names) == "p"
    assert render.preferred_field([]) == ""


def test_prepare_render_mesh_scalar_and_solid(demo) -> None:
    # 벡터 필드는 magnitude 스칼라로 컬러링된다.
    mesh, scalar = render.prepare_render_mesh(demo, "U", timestep=2)
    assert scalar == "U__mag"
    assert scalar in mesh.point_data
    # 데모 격자는 z 두께가 0 인 평면이다.
    assert render.mesh_is_flat(mesh)

    # 빈 field 는 solid color (scalar 없음).
    _, empty_scalar = render.prepare_render_mesh(demo, "", timestep=0)
    assert empty_scalar == ""


def test_prepare_render_mesh_scalar_field(demo) -> None:
    mesh, scalar = render.prepare_render_mesh(demo, "p", timestep=0)
    assert scalar == "p"
    assert scalar in mesh.point_data


# ──────────────────────────────────────────────────────────────────────
# 파일 브라우저 — list_directory
# ──────────────────────────────────────────────────────────────────────


def test_list_directory_lists_dirs_and_loadable_files(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "case.vtk").write_text("x")
    (tmp_path / "proj.ntwin").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("skip me")  # 로드 불가 확장자
    (tmp_path / ".hidden").write_text("skip")  # 숨김

    result = service.list_directory(tmp_path)
    names = {e["name"] for e in result["entries"]}
    assert "sub" in names
    assert "case.vtk" in names
    assert "proj.ntwin" in names
    assert "notes.txt" not in names
    assert ".hidden" not in names
    # 디렉토리가 파일보다 먼저 정렬된다.
    assert result["entries"][0]["is_dir"] is True
    assert result["parent"] == str(tmp_path.parent)
    assert result["cwd"] == str(tmp_path)


def test_list_directory_file_path_falls_back_to_parent(tmp_path) -> None:
    f = tmp_path / "case.vtu"
    f.write_text("x")
    result = service.list_directory(f)
    assert result["cwd"] == str(tmp_path)


def test_list_directory_missing_path_falls_back_home(tmp_path) -> None:
    result = service.list_directory(tmp_path / "does-not-exist")
    from pathlib import Path

    assert result["cwd"] == str(Path.home())


# ──────────────────────────────────────────────────────────────────────
# Physics AI — NVIDIA PhysicsNeMo 스타일 직접 필드 예측
# ──────────────────────────────────────────────────────────────────────


def test_build_physics_ai_twin_trains_and_predicts(demo) -> None:
    result = service.build_physics_ai_twin(demo, "p", hidden=8, max_epochs=5, max_train_points=500)
    assert result["param_min"] == pytest.approx(0.0)
    assert result["param_max"] > result["param_min"]
    assert "rmse" in result["validation_metrics"]
    assert len(result["train_losses"]) == 5

    engine = result["engine"]
    assert engine.reducer_type == "direct_physics_ai"
    mid = 0.5 * (result["param_min"] + result["param_max"])
    prediction = service.predict_twin(engine, mid)
    assert prediction.shape[0] == demo.n_points

    # attach_prediction / save_engine 은 TwinEngine 과 동일한 계약으로 동작한다.
    field_name = service.attach_prediction(demo, prediction, field_name="physics_pred")
    assert field_name in demo.mesh.point_data


def test_build_physics_ai_twin_requires_multiple_timesteps() -> None:
    single = service.make_demo_dataset(nx=10, ny=10, n_steps=1)
    with pytest.raises(ValueError):
        service.build_physics_ai_twin(single, "p")


def test_export_physicsnemo_module_requires_trained_model(demo) -> None:
    class _Empty:
        model = None

    with pytest.raises(RuntimeError):
        service.export_physicsnemo_module(_Empty(), "/tmp/nonexistent_physicsnemo.pt")


def test_export_physicsnemo_module(demo, tmp_path) -> None:
    """physicsnemo 설치 여부에 따라 실제 내보내기 또는 안내 에러를 검증한다."""
    result = service.build_physics_ai_twin(demo, "p", hidden=8, max_epochs=3, max_train_points=200)
    engine = result["engine"]
    from naviertwin.core.physnemo.physicsnemo_model import physicsnemo_available

    out = tmp_path / "module.pt"
    if physicsnemo_available():
        path = service.export_physicsnemo_module(engine, out)
        assert Path(path).exists()
    else:
        with pytest.raises(RuntimeError, match="physicsnemo"):
            service.export_physicsnemo_module(engine, out)


def test_recommend_method_rom_for_small_series(demo) -> None:
    rec = service.recommend_method(demo)
    assert rec["method"] == "rom"
    assert "ROM" in rec["reason"]


def test_recommend_method_none_for_single_step() -> None:
    single = service.make_demo_dataset(nx=8, ny=8, n_steps=1)
    rec = service.recommend_method(single)
    assert rec["method"] == "none"


def test_compare_models_includes_physics_row_for_scalar_field(demo) -> None:
    """스칼라 필드 비교에는 Physics AI 행이 포함되고, RMSE 순위에 함께 선다."""
    result = service.compare_models(
        demo, "p", 3, combos=[("pod", "rbf")], physics_epochs=3
    )
    combos = [r["combo"] for r in result["rows"]]
    assert "pod+rbf" in combos
    assert any("physicsnemo" in c for c in combos)
    physics_row = next(r for r in result["rows"] if "physicsnemo" in r["combo"])
    assert physics_row["status"] == "ok"
    assert physics_row["n_modes"] == 0  # 모드 개념 없음


def test_compare_models_includes_physics_for_vector_field_via_magnitude(demo) -> None:
    """벡터 필드도 양쪽 모두 크기(magnitude) 스냅샷으로 학습하므로 동일 조건 비교."""
    result = service.compare_models(
        demo, "U", 3, combos=[("pod", "rbf")], physics_epochs=3
    )
    combos = [r["combo"] for r in result["rows"]]
    assert "pod+rbf" in combos
    assert any("physicsnemo" in c for c in combos)


def test_build_physics_ai_twin_multi_output(demo) -> None:
    """다중 출력: 한 신경망이 p 와 U(크기)를 동시 학습하고 필드별로 분해된다."""
    result = service.build_physics_ai_twin(
        demo, ["p", "U"], hidden=8, max_epochs=3, max_train_points=300
    )
    assert result["fields"] == ["p", "U"]
    engine = result["engine"]
    n_points = demo.n_points

    mid = 0.5 * (result["param_min"] + result["param_max"])
    prediction = service.predict_twin(engine, mid)
    assert prediction.shape[0] == n_points * 2  # field-major 로 이어붙은 벡터

    parts = service.split_multi_prediction(engine, prediction)
    assert parts is not None
    names = [name for name, _ in parts]
    assert names == ["p", "U_mag"]  # 벡터 U 는 크기(magnitude)로 학습
    for _, segment in parts:
        assert segment.shape[0] == n_points

    # 필드별 검증 지표도 딸려 온다.
    assert set(result["per_field_metrics"].keys()) == {"p", "U"}


def test_split_multi_prediction_returns_none_for_single_output(demo) -> None:
    """단일 출력 엔진(고전 TwinEngine 포함)은 기존 단일 attach 경로를 유지한다."""
    rom = service.build_twin(demo, "p", n_modes=3)["engine"]
    assert service.split_multi_prediction(rom, np.zeros(4)) is None
    single = service.build_physics_ai_twin(demo, "p", hidden=8, max_epochs=2)["engine"]
    prediction = service.predict_twin(single, 0.5)
    assert service.split_multi_prediction(single, prediction) is None


# ──────────────────────────────────────────────────────────────────────
# 케이스 세트 — 정상 파라미터 스윕 (문제 유형 B)
# ──────────────────────────────────────────────────────────────────────


def _write_case_set(directory, *, n_cases=4, with_csv=True):
    """정상해 케이스 N개(.vtk) + 파라미터 CSV 를 폴더에 쓴다.

    각 케이스는 동일 격자, p = inlet 속도에 선형 비례하는 장 → 스윕 학습이
    정확히 복원 가능해야 한다.
    """
    import pyvista as pv

    directory.mkdir(parents=True, exist_ok=True)
    velocities = [1.0 + i for i in range(n_cases)]
    for index, velocity in enumerate(velocities):
        grid = pv.ImageData(dimensions=(5, 5, 1))
        coords = np.asarray(grid.points, dtype=float)
        grid.point_data["p"] = velocity * (coords[:, 0] + 2.0 * coords[:, 1])
        grid.save(directory / f"case_{index:02d}.vtk")
    if with_csv:
        rows = ["inlet_velocity"] + [f"{v}" for v in velocities]
        (directory / "params.csv").write_text("\n".join(rows) + "\n")
    return velocities


def test_load_case_set_with_param_csv(tmp_path) -> None:
    velocities = _write_case_set(tmp_path / "sweep", n_cases=4)
    result = service.load_case_set(tmp_path / "sweep")

    assert len(result["datasets"]) == 4
    assert result["param_names"] == ["inlet_velocity"]
    assert result["params"].shape == (4, 1)
    np.testing.assert_allclose(result["params"][:, 0], velocities)
    assert result["params_source"].startswith("CSV:")
    assert result["case_names"] == [f"case_{i:02d}.vtk" for i in range(4)]
    # 각 케이스는 단일 스냅샷으로 materialize 된다 (params 행 수와 1:1).
    assert all(int(ds.n_time_steps) == 1 for ds in result["datasets"])


def test_load_case_set_without_csv_falls_back_to_case_index(tmp_path) -> None:
    _write_case_set(tmp_path / "nocsv", n_cases=3, with_csv=False)
    result = service.load_case_set(tmp_path / "nocsv")
    assert result["param_names"] == ["case_index"]
    np.testing.assert_allclose(result["params"][:, 0], [0.0, 1.0, 2.0])
    assert "case_index" in result["params_source"]


def test_load_case_set_requires_two_cases(tmp_path) -> None:
    _write_case_set(tmp_path / "one", n_cases=1, with_csv=False)
    with pytest.raises(ValueError, match="2개 이상"):
        service.load_case_set(tmp_path / "one")


def test_load_case_set_rejects_row_count_mismatch(tmp_path) -> None:
    _write_case_set(tmp_path / "bad", n_cases=3, with_csv=False)
    (tmp_path / "bad" / "params.csv").write_text("inlet_velocity\n1.0\n2.0\n")  # 3개 중 2행
    with pytest.raises(ValueError):
        service.load_case_set(tmp_path / "bad")


def test_build_twin_from_cases_predicts_swept_parameter(tmp_path) -> None:
    """스윕 ROM: 학습한 운전조건에서 해당 케이스 장을 재현해야 한다."""
    _write_case_set(tmp_path / "sweep", n_cases=4)
    loaded = service.load_case_set(tmp_path / "sweep")
    result = service.build_twin_from_cases(
        loaded["datasets"], "p", 3, loaded["params"], param_names=loaded["param_names"]
    )
    assert result["n_cases"] == 4
    assert result["param_names"] == ["inlet_velocity"]
    assert result["param_mins"] == [1.0]
    assert result["param_maxs"] == [4.0]
    meta = result["engine"].training_metadata
    assert meta["problem_type"] == "steady_sweep"

    truth = np.asarray(loaded["datasets"][1].extract_field_snapshots("p"))[:, -1]
    prediction = service.predict_twin(result["engine"], [2.0])  # 케이스 #1 조건
    assert prediction.shape == truth.shape
    np.testing.assert_allclose(prediction, truth, rtol=1e-3, atol=1e-3)


def test_build_twin_from_cases_rejects_param_row_mismatch(tmp_path) -> None:
    _write_case_set(tmp_path / "sweep", n_cases=3)
    loaded = service.load_case_set(tmp_path / "sweep")
    with pytest.raises(ValueError, match="파라미터 행 수"):
        service.build_twin_from_cases(loaded["datasets"], "p", 2, np.zeros((2, 1)))


def test_build_physics_ai_twin_from_cases(tmp_path) -> None:
    _write_case_set(tmp_path / "sweep", n_cases=4)
    loaded = service.load_case_set(tmp_path / "sweep")
    result = service.build_physics_ai_twin_from_cases(
        loaded["datasets"],
        "p",
        loaded["params"],
        param_names=loaded["param_names"],
        hidden=8,
        max_epochs=3,
    )
    assert result["n_cases"] == 4
    assert result["param_names"] == ["inlet_velocity"]
    assert result["param_mins"] == [1.0]
    assert result["engine"].training_metadata["problem_type"] == "steady_sweep"
    prediction = service.predict_twin(result["engine"], [2.5])
    assert prediction.shape[0] == loaded["datasets"][0].n_points


# ──────────────────────────────────────────────────────────────────────
# 출력 격자 자유화 (M3) — 신경장은 학습 격자에 묶이지 않는다
# ──────────────────────────────────────────────────────────────────────


def test_predict_at_arbitrary_coords_is_resolution_free(tmp_path) -> None:
    """학습 격자보다 촘촘한 격자에서 평가해도 참값을 따라가야 한다."""
    import pyvista as pv

    _write_case_set(tmp_path / "sweep", n_cases=5)
    loaded = service.load_case_set(tmp_path / "sweep")
    result = service.build_physics_ai_twin_from_cases(
        loaded["datasets"],
        "p",
        loaded["params"],
        param_names=loaded["param_names"],
        hidden=48,
        max_epochs=400,
    )
    model = result["engine"].model

    # 학습 격자는 5×5, 평가 격자는 9×9 (학습에 없던 좌표 포함).
    fine = pv.ImageData(dimensions=(9, 9, 1), spacing=(4 / 8, 4 / 8, 1))
    coords = np.asarray(fine.points, dtype=float)[:, :3]
    velocity = 3.0  # 학습된 케이스 조건 중 하나
    values = np.asarray(model.predict_at(coords, [velocity]), dtype=float).reshape(-1)
    assert values.shape[0] == coords.shape[0]  # 학습 격자(25) 가 아니라 81 점

    truth = velocity * (coords[:, 0] + 2.0 * coords[:, 1])
    # 신경망 근사라 느슨한 허용오차 — 값의 스케일 대비 상대 오차로 본다.
    rel = np.linalg.norm(values - truth) / np.linalg.norm(truth)
    assert rel < 0.15, f"미세 격자 예측이 참값에서 너무 벗어남: rel={rel:.3f}"


def test_predict_to_mesh_attaches_fields_on_target(tmp_path) -> None:
    """대상 메쉬에 예측 필드를 붙인 새 데이터셋을 돌려준다 (원본 불변)."""
    import pyvista as pv

    _write_case_set(tmp_path / "sweep", n_cases=4)
    loaded = service.load_case_set(tmp_path / "sweep")
    engine = service.build_physics_ai_twin_from_cases(
        loaded["datasets"], ["p"], loaded["params"],
        param_names=loaded["param_names"], hidden=8, max_epochs=3,
    )["engine"]

    fine = pv.ImageData(dimensions=(7, 7, 1), spacing=(4 / 6, 4 / 6, 1))
    fine.save(tmp_path / "target.vtk")
    target = service.load_dataset(tmp_path / "target.vtk")

    dataset, attached = service.predict_to_mesh(engine, [2.0], target)
    assert attached == ["twin_p"]
    assert dataset.n_points == 49  # 학습 격자(25) 와 다른 해상도
    assert "twin_p" in dataset.mesh.point_data
    assert np.isfinite(dataset.mesh.point_data["twin_p"]).all()
    # 원본 target 은 건드리지 않는다.
    assert "twin_p" not in target.mesh.point_data


def test_predict_to_mesh_rejects_rom_engine(demo) -> None:
    """ROM 은 POD 모드가 학습 메쉬에 묶여 임의 격자 예측이 불가하다."""
    rom = service.build_twin(demo, "p", n_modes=3)["engine"]
    with pytest.raises(RuntimeError, match="Physics AI"):
        service.predict_to_mesh(rom, [0.5], demo)


def test_predict_at_rejects_bad_shapes(demo) -> None:
    model = service.build_physics_ai_twin(
        demo, "p", hidden=8, max_epochs=2
    )["engine"].model
    with pytest.raises(ValueError, match="coords"):
        model.predict_at(np.zeros((4, 2)), [0.5])  # 3D 좌표가 아님
    with pytest.raises(ValueError, match="parameter dimension"):
        model.predict_at(np.zeros((4, 3)), [0.5, 1.5])  # 파라미터 1개인 모델


# ──────────────────────────────────────────────────────────────────────
# 형상 가변 (M4a) — 공통 격자 재샘플 + SDF
# ──────────────────────────────────────────────────────────────────────


def _write_varying_geometry_cases(directory, *, radii=(0.10, 0.15, 0.20, 0.25)):
    """반지름이 다른 원기둥 구멍을 뚫은 케이스들 — 케이스마다 메쉬가 다르다.

    2D 원기둥 주위 유동의 형상 스윕 축소판. threshold 로 구멍을 뚫으므로
    반지름마다 점/셀 수가 달라진다(= 동일 메쉬 제약 위반).
    """
    import pyvista as pv

    directory.mkdir(parents=True, exist_ok=True)
    for index, radius in enumerate(radii):
        grid = pv.ImageData(dimensions=(21, 21, 1), spacing=(1 / 20, 1 / 20, 1))
        coords = np.asarray(grid.points, dtype=float)
        dist = np.linalg.norm(coords[:, :2] - 0.5, axis=1)
        ug = grid.cast_to_unstructured_grid()
        ug.point_data["dist"] = dist
        # 원 밖만 남긴다 → 반지름마다 다른 메쉬.
        case = ug.threshold(radius, scalars="dist")
        pts = np.asarray(case.points, dtype=float)
        d = np.linalg.norm(pts[:, :2] - 0.5, axis=1)
        # 반지름이 클수록 강해지는 합성 압력장 (형상 → 물리 의존성).
        case.point_data["p"] = (1.0 / np.maximum(d, 1e-3)) * radius
        case.save(directory / f"case_{index:02d}.vtu")
    rows = ["radius"] + [f"{r}" for r in radii]
    (directory / "params.csv").write_text("\n".join(rows) + "\n")
    return list(radii)


def test_load_case_set_keeps_varying_geometry_when_resample_false(tmp_path) -> None:
    """resample=False 면 메쉬가 달라도 그대로 둔다 (진짜 구멍 데이터 대응).

    예전에는 무조건 거부했지만, 격자에 진짜 구멍이 뚫린 CFD 결과는 공통 격자로
    옮기는 순간 구멍이 가짜 empty 가 되므로 "재샘플 안 함" 이 선택 가능해야 한다.
    대신 점 수가 달라 ROM 은 불가하고 Physics AI 만 학습할 수 있다.
    """
    _write_varying_geometry_cases(tmp_path / "shapes")
    result = service.load_case_set(tmp_path / "shapes", resample=False)
    assert result["resampled"] is False
    assert "재샘플 안 함" in result["grid_summary"]
    assert len({ds.n_points for ds in result["datasets"]}) > 1
    # sdf 를 만들지 않는다 — 진짜 구멍을 쓰겠다는 선택이므로.
    for ds in result["datasets"]:
        assert "sdf" not in ds.mesh.point_data


def test_load_case_set_auto_resamples_varying_geometry(tmp_path) -> None:
    """형상이 다른 케이스는 auto 로 공통 격자에 올라간다 (문제 유형 B 로 환원)."""
    radii = _write_varying_geometry_cases(tmp_path / "shapes")
    result = service.load_case_set(tmp_path / "shapes", resolution=16)

    assert result["resampled"] is True
    assert "공통 격자" in result["grid_summary"]
    np.testing.assert_allclose(result["params"][:, 0], radii)
    datasets = result["datasets"]
    # 재샘플 후에는 모든 케이스가 같은 격자를 공유한다 — 이게 학습의 전제.
    assert service.meshes_are_identical(datasets)
    assert len({ds.n_points for ds in datasets}) == 1
    for ds in datasets:
        assert "sdf" in ds.mesh.point_data
        assert "p" in ds.mesh.point_data


def test_resample_sdf_sign_tracks_geometry(tmp_path) -> None:
    """sdf 는 유체에서 +, 고체(구멍) 안에서 − 이고 반지름이 커지면 고체가 넓어진다."""
    _write_varying_geometry_cases(tmp_path / "shapes", radii=(0.10, 0.30))
    result = service.load_case_set(tmp_path / "shapes", resolution=24)
    small, large = result["datasets"]

    center_small = np.asarray(small.mesh.point_data["sdf"])
    center_large = np.asarray(large.mesh.point_data["sdf"])
    coords = np.asarray(small.mesh.points, dtype=float)
    at_center = np.argmin(np.linalg.norm(coords[:, :2] - 0.5, axis=1))
    # 도메인 중앙은 두 경우 모두 구멍 안 → 음수(고체).
    assert center_small[at_center] < 0
    assert center_large[at_center] < 0
    # 반지름이 크면 고체 영역(sdf<0)이 더 넓다.
    assert int((center_large < 0).sum()) > int((center_small < 0).sum())


def test_varying_geometry_end_to_end_rom(tmp_path) -> None:
    """형상 가변 케이스로 ROM 스윕 학습 → 형상 파라미터로 예측까지 성립한다."""
    _write_varying_geometry_cases(tmp_path / "shapes")
    loaded = service.load_case_set(tmp_path / "shapes", resolution=16)
    result = service.build_twin_from_cases(
        loaded["datasets"], "p", 3, loaded["params"],
        param_names=loaded["param_names"],
    )
    assert result["param_names"] == ["radius"]
    prediction = service.predict_twin(result["engine"], [0.175])  # 학습 반지름 사이
    assert prediction.shape[0] == loaded["datasets"][0].n_points
    assert np.isfinite(prediction).all()


def test_resample_is_skipped_when_meshes_match(tmp_path) -> None:
    """메쉬가 같으면 auto 는 재샘플하지 않는다 (불필요한 보간 오차 방지)."""
    _write_case_set(tmp_path / "same", n_cases=3)
    result = service.load_case_set(tmp_path / "same")
    assert result["resampled"] is False
    assert result["grid_summary"] == ""


# ──────────────────────────────────────────────────────────────────────
# 계열 Ⓓ — DMD 동역학 예보 (PyDMD)
# ──────────────────────────────────────────────────────────────────────


def _dmd_friendly_dataset(n_steps=48):
    """DMD 가 이론적으로 정확히 맞추는 데이터 — 공간모드 2개 × 고유 주파수.

    각 모드가 자기 주파수/성장률을 갖는 저랭크 선형 동역학(진행파). 실수
    데이터라 켤레쌍 때문에 실제 랭크는 4 다.
    """
    import pyvista as pv

    from naviertwin.core.cfd_reader.base import CFDDataset

    nx, ny = 24, 12
    grid = pv.ImageData(dimensions=(nx, ny, 1), spacing=(2 * np.pi / (nx - 1), 1.0, 1))
    coords = np.asarray(grid.points, dtype=float)
    x = coords[:, 0]
    times = np.linspace(0.0, 6.0, n_steps)
    series = np.stack(
        [
            np.exp(-0.10 * t) * np.sin(x - 1.3 * t)
            + np.exp(-0.30 * t) * np.sin(2 * x - 2.7 * t)
            for t in times
        ],
        axis=0,
    )
    grid.point_data["p"] = series[0]
    return CFDDataset(
        mesh=grid,
        time_steps=[float(t) for t in times],
        field_names=["p"],
        metadata={
            "source": "dmd_friendly",
            "time_series_fields": {"p": series},
            "time_series_locations": {"p": "point"},
        },
    )


def test_build_dmd_twin_reconstructs_and_reports_fit(_=None) -> None:
    """적합한 데이터에서 DMD 는 기계 정밀도로 재구성하고 주파수를 복원한다."""
    ds = _dmd_friendly_dataset()
    result = service.build_dmd_twin(ds, "p")

    assert result["method"] == "dmd"
    assert result["reconstruction_error"] < 1e-6
    assert result["forecast_max"] > result["param_max"]  # 외삽 여유
    # 참 주파수 1.3/2pi=0.207, 2.7/2pi=0.430 를 복원한다.
    freqs = sorted({round(abs(f), 3) for f in result["frequencies"]})
    assert any(abs(f - 0.207) < 0.02 for f in freqs), freqs
    assert any(abs(f - 0.430) < 0.02 for f in freqs), freqs
    meta = result["engine"].training_metadata
    assert meta["problem_type"] == "dynamics_forecast"


def test_dmd_twin_forecasts_beyond_training_window() -> None:
    """계열 Ⓓ의 존재 이유 — 학습에 없던 미래 시간을 실제로 맞춘다."""
    full = _dmd_friendly_dataset(n_steps=48)
    series = np.asarray(full.metadata["time_series_fields"]["p"])
    times = np.asarray(full.time_steps)

    # 앞 절반만 담은 데이터셋으로 학습 → 뒤 절반은 완전히 미지의 구간.
    from naviertwin.core.cfd_reader.base import CFDDataset

    half = 24
    train = CFDDataset(
        mesh=full.mesh.copy(deep=True),
        time_steps=[float(t) for t in times[:half]],
        field_names=["p"],
        metadata={
            "source": "dmd_friendly",
            "time_series_fields": {"p": series[:half]},
            "time_series_locations": {"p": "point"},
        },
    )
    engine = service.build_dmd_twin(train, "p")["engine"]

    future_t = float(times[-1])  # 학습 구간 밖
    truth = series[-1]
    pred = service.predict_twin(engine, future_t)
    rel = float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))
    assert rel < 1e-3, f"미래 외삽이 부정확: rel={rel}"


def test_build_dmd_twin_reports_poor_fit_on_unsuitable_data() -> None:
    """부적합 데이터(불연속 필라멘트)에서는 재구성 오차가 크게 보고된다.

    DMD 는 안 맞아도 조용히 틀리므로, UI 가 신뢰도를 보여줄 수 있어야 한다.
    """
    ds = service.make_demo_dataset(nx=16, ny=16, n_steps=12, kind="filament")
    result = service.build_dmd_twin(ds, "p")
    assert result["reconstruction_error"] > 0.1  # 눈에 띄게 나쁨 → UI 경고 대상


def test_build_dmd_twin_rejects_bad_input() -> None:
    single = service.make_demo_dataset(nx=8, ny=8, n_steps=1)
    with pytest.raises(ValueError, match="2개 이상"):
        service.build_dmd_twin(single, "p")
    ds = service.make_demo_dataset(nx=8, ny=8, n_steps=6)
    with pytest.raises(ValueError, match="지원하지 않는"):
        service.build_dmd_twin(ds, "p", method="fbdmd")  # 검증 안 된 변형은 차단


# ──────────────────────────────────────────────────────────────────────
# 해상도 낮추기 (coarsen) — 대용량 메모리 대응
# ──────────────────────────────────────────────────────────────────────


def test_coarsen_dataset_reduces_points_and_preserves_series() -> None:
    """성긴 격자로 재샘플해도 시계열·필드·물리 범위가 보존된다."""
    ds = service.make_demo_dataset(nx=80, ny=80, n_steps=8, kind="advecting")
    result = service.coarsen_dataset(ds, resolution=24)
    out = result["dataset"]

    assert result["points_after"] < result["points_before"]
    assert result["ratio"] > 3.0
    assert "→" in result["summary"]
    # 시계열과 벡터장이 모두 살아남는다.
    assert out.n_time_steps == ds.n_time_steps
    assert set(out.field_names) == {"U", "p"}
    assert out.metadata["time_series_fields"]["U"].shape[2] == 3

    full = ds.extract_field_snapshots("p")
    small = out.extract_field_snapshots("p")
    assert small.shape[1] == full.shape[1]  # 스텝 수 동일
    assert small.nbytes < full.nbytes / 3  # 메모리 실제 절감
    # 저해상도지만 물리 범위는 비슷하게 유지 (손실 압축이므로 여유 허용).
    assert abs(float(small.max()) - float(full.max())) < 0.1 * float(full.max() - full.min())
    # 시간 진화가 뭉개지지 않는다.
    assert not np.allclose(small[:, 0], small[:, -1])
    # 원본은 변경되지 않는다.
    assert ds.n_points == result["points_before"]


def test_estimate_coarsen_matches_actual_coarsen() -> None:
    """미리보기는 실제 재샘플 결과와 정확히 일치해야 한다.

    사용자는 이 숫자를 보고 해상도를 고르므로, 추정이 실제와 다르면 미리보기가
    있으나 마나다. 여러 해상도에서 점 수가 정확히 같은지 확인한다.
    """
    ds = service.make_demo_dataset(nx=60, ny=60, n_steps=4, kind="advecting")
    for resolution in (12, 24, 48):
        estimate = service.estimate_coarsen(ds, resolution)
        actual = service.coarsen_dataset(ds, resolution=resolution)
        assert estimate["points_after"] == actual["points_after"]
        assert estimate["points_before"] == actual["points_before"]
        assert estimate["ratio"] == pytest.approx(actual["ratio"])


def test_estimate_coarsen_is_monotonic_and_reports_memory() -> None:
    """해상도를 올리면 점이 늘고, 스냅샷 행렬 추정은 실제 바이트와 맞는다."""
    ds = service.make_demo_dataset(nx=60, ny=60, n_steps=5, kind="advecting")
    low = service.estimate_coarsen(ds, 16)
    high = service.estimate_coarsen(ds, 64)

    assert low["points_after"] < high["points_after"]
    assert low["ratio"] > high["ratio"]  # 성길수록 더 많이 줄어든다
    assert low["bytes_after"] < low["bytes_before"]
    # 요약은 사람이 읽을 형태 — 치수·축소비·메모리가 모두 들어간다.
    assert "×" in low["summary"] and "점" in low["summary"]
    assert "→" in low["summary"]
    # 메모리 추정이 실제 스냅샷 행렬과 일치한다 (점 × 스텝 × float64).
    coarse = service.coarsen_dataset(ds, resolution=16)["dataset"]
    assert low["bytes_after"] == coarse.extract_field_snapshots("p").nbytes


def test_coarsen_dataset_rejects_when_no_base_fields() -> None:
    import pyvista as pv

    from naviertwin.core.cfd_reader.base import CFDDataset

    grid = pv.ImageData(dimensions=(4, 4, 1))
    empty = CFDDataset(mesh=grid, time_steps=[0.0], field_names=[], metadata={})
    with pytest.raises(ValueError, match="field"):
        service.coarsen_dataset(empty)


# ──────────────────────────────────────────────────────────────────────
# 입력 필드 (field-to-field 연산자)
# ──────────────────────────────────────────────────────────────────────


def _field_to_field_dataset(n_steps: int = 12, n_side: int = 12):
    """p = 2·s 인 합성 데이터 — 입력 field s 가 출력을 결정한다.

    s 는 좌표·시간과 무관한 난수라, 입력 field 없이 (x, t) 만으로는 학습이
    불가능하다 → 입력 field 경로가 실제로 쓰이는지 구분할 수 있다.
    """
    import pyvista as pv

    from naviertwin.core.cfd_reader.base import CFDDataset

    rng = np.random.default_rng(0)
    grid = pv.ImageData(dimensions=(n_side, n_side, 1), spacing=(1 / (n_side - 1),) * 2 + (1,))
    s = rng.random((n_steps, grid.n_points))
    p = 2.0 * s
    grid.point_data["s"] = s[0]
    grid.point_data["p"] = p[0]
    return CFDDataset(
        mesh=grid,
        time_steps=[float(i) for i in range(n_steps)],
        field_names=["s", "p"],
        metadata={
            "source": "field_to_field",
            "time_series_fields": {"s": s, "p": p},
            "time_series_locations": {"s": "point", "p": "point"},
        },
    ), s, p


def test_input_fields_make_a_field_to_field_operator() -> None:
    """입력 field 를 주면 학습에 없던 새 입력장에도 적용되는 연산자가 된다."""
    ds, _s, _p = _field_to_field_dataset()
    result = service.build_physics_ai_twin(
        ds, "p", input_fields=["s"], hidden=32, max_epochs=600
    )
    assert result["input_fields"] == ["s"]
    model = result["engine"].model
    assert model.n_input_features == 1

    # 학습에 전혀 없던 새 입력장 → p = 2s 를 재현해야 한다.
    rng = np.random.default_rng(99)
    coords = np.asarray(ds.mesh.points, dtype=float)[:, :3]
    s_new = rng.random((1, ds.n_points, 1))
    pred = model.predict_at(coords, np.array([[5.0]]), input_features=s_new).reshape(-1)
    truth = 2.0 * s_new.reshape(-1)
    rel = float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))
    assert rel < 0.15, f"새 입력장 예측이 부정확: rel={rel}"


def test_without_input_fields_random_field_is_unlearnable() -> None:
    """대조군 — 입력 field 없이 (x,t) 만으로는 난수 기반 출력을 못 맞춘다."""
    ds, _s, _p = _field_to_field_dataset()
    with_input = service.build_physics_ai_twin(
        ds, "p", input_fields=["s"], hidden=32, max_epochs=600
    )["validation_metrics"]["rmse"]
    without = service.build_physics_ai_twin(ds, "p", hidden=32, max_epochs=600)[
        "validation_metrics"
    ]["rmse"]
    assert with_input < without / 3, f"입력 field 효과 없음: {with_input} vs {without}"


def test_predict_twin_uses_stored_input_fields(_=None) -> None:
    """③Twin 은 학습 입력장을 시간 보간해 자동으로 채운다 (슬라이더 그대로 동작)."""
    ds, _s, _p = _field_to_field_dataset()
    engine = service.build_physics_ai_twin(
        ds, "p", input_fields=["s"], hidden=32, max_epochs=400
    )["engine"]
    pred = service.predict_twin(engine, 3.0)  # 학습 시간 지점
    assert pred.shape[0] == ds.n_points
    assert np.isfinite(pred).all()


def test_input_and_output_field_overlap_is_rejected() -> None:
    ds, _s, _p = _field_to_field_dataset(n_steps=4, n_side=6)
    with pytest.raises(ValueError, match="같은 field"):
        service.build_physics_ai_twin(
            ds, "p", input_fields=["p"], hidden=8, max_epochs=2
        )


def test_predict_at_requires_input_features_when_trained_with_them() -> None:
    ds, _s, _p = _field_to_field_dataset(n_steps=4, n_side=6)
    model = service.build_physics_ai_twin(
        ds, "p", input_fields=["s"], hidden=8, max_epochs=2
    )["engine"].model
    coords = np.asarray(ds.mesh.points, dtype=float)[:, :3]
    with pytest.raises(ValueError, match="입력 field"):
        model.predict_at(coords, np.array([[1.0]]))


# ──────────────────────────────────────────────────────────────────────
# 데모 카탈로그 — 계열별로 "잘 맞는 데이터"가 하나씩 있어야 한다
# ──────────────────────────────────────────────────────────────────────


def test_demo_catalog_covers_both_problem_types() -> None:
    values = [d["value"] for d in service.DEMO_CATALOG]
    assert set(values) == set(service.DEMO_TIME_SERIES_KINDS) | set(
        service.DEMO_CASE_SET_KINDS
    )
    for entry in service.DEMO_CATALOG:
        assert entry["title"] and entry["note"]  # UI 가 그대로 보여준다


def test_waves_demo_is_dmd_friendly() -> None:
    """이 데모의 존재 이유 — DMD 가 실제로 맞는 유일한 데이터."""
    ds = service.make_demo_dataset(nx=24, ny=24, n_steps=32, kind="waves")
    assert service.dataset_info(ds)["source"] == "demo_traveling_waves"
    result = service.build_dmd_twin(ds, "p")
    # 필라멘트 데모는 0.66 — 여기선 기계 정밀도로 맞아야 한다.
    assert result["reconstruction_error"] < 1e-6
    freqs = sorted({round(abs(f), 3) for f in result["frequencies"]})
    assert any(abs(f - 0.207) < 0.02 for f in freqs), freqs  # 1.3/2π
    assert any(abs(f - 0.430) < 0.02 for f in freqs), freqs  # 2.7/2π


def test_waves_demo_contrasts_with_filament_for_dmd() -> None:
    """대조 — 같은 DMD 로 필라멘트는 크게 빗나가야 (적합도 배지가 의미를 가짐)."""
    waves = service.build_dmd_twin(
        service.make_demo_dataset(nx=20, ny=20, n_steps=24, kind="waves"), "p"
    )["reconstruction_error"]
    filament = service.build_dmd_twin(
        service.make_demo_dataset(nx=20, ny=20, n_steps=24, kind="filament"), "p"
    )["reconstruction_error"]
    assert waves < 1e-6 < 0.1 < filament


def test_demo_case_set_sweep_shares_mesh_and_trains() -> None:
    """정상 스윕 데모: 동일 메쉬 · 2개 운전조건 → ROM 학습까지 성립."""
    result = service.make_demo_case_set("sweep", n_side=24)
    datasets = result["datasets"]
    assert len(datasets) == 5
    assert result["param_names"] == ["inlet_velocity", "angle_of_attack"]
    assert result["params"].shape == (5, 2)
    assert result["resampled"] is False
    assert service.meshes_are_identical(datasets)

    twin = service.build_twin_from_cases(
        datasets, "p", 3, result["params"], param_names=result["param_names"]
    )
    pred = service.predict_twin(twin["engine"], [17.5, 3.0])  # 학습 조건 사이
    assert pred.shape[0] == datasets[0].n_points
    assert np.isfinite(pred).all()


def test_demo_case_set_shapes_is_resampled_with_sdf() -> None:
    """형상 가변 데모: 메쉬가 달라 공통 격자로 재샘플되고 sdf 가 붙는다."""
    result = service.make_demo_case_set("shapes", n_side=32, resolution=24)
    datasets = result["datasets"]
    assert len(datasets) == 5
    assert result["param_names"] == ["radius"]
    assert result["resampled"] is True
    assert "공통 격자" in result["grid_summary"]
    # 재샘플 후에는 학습이 가능하도록 메쉬가 통일된다.
    assert service.meshes_are_identical(datasets)
    for ds in datasets:
        assert "sdf" in ds.mesh.point_data
    # 반지름이 클수록 고체 영역(sdf<0)이 넓다 — 형상이 실제로 다르다는 증거.
    solid_small = int((np.asarray(datasets[0].mesh.point_data["sdf"]) < 0).sum())
    solid_large = int((np.asarray(datasets[-1].mesh.point_data["sdf"]) < 0).sum())
    assert solid_large > solid_small


def test_make_demo_case_set_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="지원하지 않는"):
        service.make_demo_case_set("bogus")
