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
