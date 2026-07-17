"""trame 웹 앱 테스트.

trame state/controller 배선과 워크플로우 콜백을 GL 컨텍스트 없이 검증한다
(``build_ui=False``). 전체 UI 빌드(PyVista 뷰어)는 GL 이 필요하므로 가능한
환경에서만 실행하고 그 외에는 skip 한다.
"""

from __future__ import annotations

import pytest

pytest.importorskip("trame", reason="웹 앱 테스트에는 trame 이 필요합니다.")
pytest.importorskip("pyvista", reason="웹 앱 테스트에는 pyvista 가 필요합니다.")


def _make_app(name: str):
    """이름이 격리된 trame server 로 앱을 생성한다 (state 누수 방지)."""
    from trame.app import get_server

    from naviertwin.web.app import NavierTwinWebApp

    server = get_server(name, client_type="vue3")
    return NavierTwinWebApp(server=server)


def test_state_and_controller_wired() -> None:
    app = _make_app("nt-test-wiring")
    st = app.server.state
    for key in [
        "nt_status",
        "nt_field",
        "nt_cmap",
        "nt_cmaps",
        "nt_n_modes",
        "nt_method",
        "nt_twin_param",
        "nt_has_dataset",
    ]:
        assert hasattr(st, key), f"missing state key: {key}"
    for cb in [
        "nt_load_demo",
        "nt_load_path",
        "nt_run_analysis",
        "nt_run_pod",
        "nt_build_twin",
        "nt_predict",
        "nt_reset_view",
    ]:
        assert callable(getattr(app.ctrl, cb, None)), f"missing controller: {cb}"


def test_load_demo_populates_state() -> None:
    app = _make_app("nt-test-demo")
    app.load_demo()
    st = app.server.state
    assert st.nt_has_dataset is True
    assert st.nt_error == ""
    assert st.nt_nsteps > 1
    assert st.nt_has_timesteps is True
    assert st.nt_field in st.nt_fields


def test_full_mvp_workflow_via_callbacks() -> None:
    """Import → Analyze → Reduce → Twin → Predict 콜백 흐름 (UI/ GL 없이)."""
    app = _make_app("nt-test-workflow")
    st = app.server.state

    app.load_demo()
    assert st.nt_has_dataset

    st.nt_method = "q_criterion"
    app.run_analysis()
    assert st.nt_error == ""
    assert "Q-criterion" in st.nt_fields

    app.run_pod()
    assert st.nt_pod_done is True
    assert st.nt_pod_summary
    assert len(st.nt_pod_energy) >= 1

    app.build_twin()
    assert st.nt_twin_ready is True
    assert st.nt_model_ready is True
    assert st.nt_twin_max > st.nt_twin_min

    st.nt_twin_param = 0.5 * (st.nt_twin_min + st.nt_twin_max)
    app.predict()
    assert st.nt_error == ""
    assert "twin_prediction" in st.nt_fields


def test_model_panel_reducer_surrogate_and_pod_mode() -> None:
    app = _make_app("nt-test-model")
    st = app.server.state
    app.load_demo()
    app.run_pod()

    # POD 모드 3D 표시
    st.nt_pod_mode = 1
    app.view_pod_mode()
    assert st.nt_error == ""
    assert "pod_mode_1" in st.nt_fields

    # Model: reducer/surrogate 선택 후에도 학습은 원본 물리량을 사용
    st.nt_reducer = "randomized_pod"
    st.nt_surrogate = "kriging"
    app.build_twin()
    assert st.nt_error == ""
    assert st.nt_model_ready is True
    assert "randomized_pod" in st.nt_model_summary
    assert "kriging" in st.nt_model_summary


def test_build_twin_dispatches_by_model_method() -> None:
    """nt_model_method='physics' 선택 시 build_twin() 이 POD 없이 직접 예측 모델을 학습한다."""
    app = _make_app("nt-test-physicsnemo")
    st = app.server.state
    app.load_demo()
    # 데모(12 스텝, 단일 시계열)는 ROM 추천 힌트가 떠야 한다.
    assert "ROM" in st.nt_method_hint

    st.nt_model_method = "physics"
    app.build_twin()
    assert st.nt_error == ""
    assert st.nt_model_ready is True
    assert st.nt_physics_ready is True
    assert st.nt_twin_ready is True
    assert "PhysicsNeMo" in st.nt_model_summary

    st.nt_twin_param = 0.5 * (st.nt_twin_min + st.nt_twin_max)
    app.predict()
    assert st.nt_error == ""
    assert "twin_prediction" in st.nt_fields

    # ROM 방식으로 다시 학습하면 physics_ready 플래그가 꺼진다.
    st.nt_model_method = "rom"
    app.build_twin()
    assert st.nt_physics_ready is False

    # operator 방식은 ⑧Bench 안내 에러를 낸다 (버튼은 UI 에서 숨겨지지만 방어).
    st.nt_model_method = "operator"
    app.build_twin()
    assert "AI Bench" in st.nt_error


def test_build_twin_legacy_physicsnemo_surrogate_shim() -> None:
    """옛 상태값(surrogate='physicsnemo')도 physics 경로로 디스패치된다."""
    app = _make_app("nt-test-physicsnemo-shim")
    st = app.server.state
    app.load_demo()
    st.nt_model_method = "rom"
    st.nt_surrogate = "physicsnemo"
    app.build_twin()
    assert st.nt_error == ""
    assert st.nt_physics_ready is True


def test_export_callbacks_write_files(tmp_path) -> None:
    import os

    app = _make_app("nt-test-export")
    st = app.server.state
    app.load_demo()
    app.run_pod()
    app.build_twin()
    st.nt_export_dir = str(tmp_path)

    # 스크린샷은 GL 플로터(빌드 UI)가 필요하므로 여기서는 제외.
    app.export_csv()
    app.export_vtk()
    app.export_project()
    app.export_engine()
    app.export_report()
    assert st.nt_error == ""
    files = set(os.listdir(tmp_path))
    assert "fields.csv" in files
    assert "engine.pkl" in files
    assert "report.html" in files
    assert any(f.endswith(".ntwin") for f in files)
    assert st.nt_export_last


def test_export_engine_requires_model() -> None:
    app = _make_app("nt-test-export-guard")
    st = app.server.state
    app.load_demo()
    app.export_engine()
    assert st.nt_error  # 모델 없음 보고


def test_fft_and_energy_charts() -> None:
    app = _make_app("nt-test-charts")
    st = app.server.state
    app.load_demo()

    app.run_fft()
    assert st.nt_error == ""
    assert st.nt_chart_dialog is True
    assert st.nt_chart_img.startswith("data:image/png;base64,")
    assert "FFT" in st.nt_chart_title
    assert st.nt_fft_summary

    # 새 차트로 갱신 (POD 에너지)
    app.run_pod()
    app.show_energy_chart()
    assert st.nt_error == ""
    assert "에너지" in st.nt_chart_title
    assert st.nt_chart_img.startswith("data:image/png;base64,")


def test_energy_chart_requires_pod() -> None:
    app = _make_app("nt-test-energy-guard")
    st = app.server.state
    app.load_demo()
    app.show_energy_chart()
    assert st.nt_error  # POD 없음 보고


def test_compare_dashboard() -> None:
    app = _make_app("nt-test-compare")
    st = app.server.state
    app.load_demo()
    app.run_compare()
    assert st.nt_error == ""
    assert st.nt_compare_dialog is True
    assert len(st.nt_compare_rows) == 4
    assert "최우수" in st.nt_compare_summary
    # 표시용 메트릭은 문자열로 포맷된다 (inf/nan JSON 안전).
    row = st.nt_compare_rows[0]
    assert isinstance(row["rmse"], str)
    assert isinstance(row["latency_ms"], str)
    assert row["status"] == "ok"


def test_compare_requires_dataset() -> None:
    app = _make_app("nt-test-compare-guard")
    st = app.server.state
    app.run_compare()
    assert st.nt_error  # 데이터 없음 보고


def test_toast_on_success_and_failure() -> None:
    app = _make_app("nt-test-toast")
    st = app.server.state
    app.load_demo()
    assert st.nt_toast_show is True
    assert st.nt_toast_color == "success"
    assert st.nt_toast_icon == "mdi-check-circle"
    # 실패 경로 → error 토스트
    st.nt_toast_show = False
    st.nt_path = "/no/such/x.vtu"
    app.load_path()
    assert st.nt_toast_show is True
    assert st.nt_toast_color == "error"


def test_pipeline_flags_and_panels() -> None:
    app = _make_app("nt-test-pipeline")
    st = app.server.state
    assert st.nt_open_panels == [0]
    assert st.nt_analysis_done is False
    app.load_demo()
    st.nt_method = "q_criterion"
    app.run_analysis()
    assert st.nt_analysis_done is True  # 파이프라인 칩 초록 조건
    # 새 데이터 로드 시 리셋
    app.load_demo()
    assert st.nt_analysis_done is False


def test_pod_max_mode_state_is_stable() -> None:
    app = _make_app("nt-test-pod-bound")
    st = app.server.state
    app.load_demo()
    st.nt_n_modes = 5
    app.run_pod()
    bound = st.nt_pod_max_mode
    assert bound >= 0
    # POD 후 모드 수를 바꿔도 모드 슬라이더 상한은 실제 계산값에 고정.
    st.nt_n_modes = 20
    assert st.nt_pod_max_mode == bound
    # 상한 초과 인덱스는 view_pod_mode 에서 clamp 되어 오류 없음.
    st.nt_pod_mode = 999
    app.view_pod_mode()
    assert st.nt_error == ""


def test_load_project_via_callback(tmp_path) -> None:
    import os

    app = _make_app("nt-test-load-proj")
    st = app.server.state
    app.load_demo()
    app.build_twin()
    st.nt_export_dir = str(tmp_path)
    app.export_project()
    ntwin = next(f for f in os.listdir(tmp_path) if f.endswith(".ntwin"))

    # 엔진/트윈 상태를 비우고 프로젝트 열기로 복원.
    app.engine = None
    st.nt_twin_ready = False
    st.nt_model_ready = False
    st.nt_path = str(tmp_path / ntwin)
    app.load_project()
    assert st.nt_error == ""
    assert st.nt_twin_ready is True
    assert st.nt_model_ready is True
    assert app.engine is not None
    # 예측 슬라이더 범위는 metadata 의 학습 범위 — 단일 스냅샷 fallback([t, t+1])
    # 으로 외삽 범위가 되면 안 된다. 데모 학습 범위는 t∈[0, 2].
    assert st.nt_twin_min == pytest.approx(0.0)
    assert st.nt_twin_max == pytest.approx(2.0)
    # 복원된 엔진으로 예측 가능.
    app.predict()
    assert st.nt_error == ""
    assert "twin_prediction" in st.nt_fields


def test_load_path_dispatches_ntwin(tmp_path) -> None:
    import os

    app = _make_app("nt-test-dispatch")
    st = app.server.state
    app.load_demo()
    st.nt_export_dir = str(tmp_path)
    app.export_project()
    ntwin = next(f for f in os.listdir(tmp_path) if f.endswith(".ntwin"))
    # load_path 가 .ntwin 확장자를 load_project 로 디스패치.
    st.nt_path = str(tmp_path / ntwin)
    app.load_path()
    assert st.nt_error == ""
    assert st.nt_has_dataset is True


def test_fft_probe_point_and_field_selection() -> None:
    app = _make_app("nt-test-fft-probe")
    st = app.server.state
    app.load_demo()

    st.nt_fft_probe = True
    st.nt_fft_point = 5
    app.run_fft()
    assert st.nt_error == ""
    assert "point[5]" in st.nt_chart_title

    # 명시적 field 선택.
    st.nt_fft_probe = False
    st.nt_fft_field = "p"
    app.run_fft()
    assert st.nt_error == ""
    assert "p" in st.nt_chart_title


def test_load_path_missing_reports_error() -> None:
    app = _make_app("nt-test-badpath")
    st = app.server.state
    st.nt_path = "/no/such/path/case.vtu"
    app.load_path()
    assert st.nt_error != ""
    assert st.nt_has_dataset is False
    # footer 중복 방지: error 는 상세만, status 가 제목을 담는다.
    assert st.nt_status == "데이터 로드 실패"
    assert not st.nt_error.startswith("데이터 로드 실패")


def test_callbacks_guard_without_dataset() -> None:
    app = _make_app("nt-test-guard")
    st = app.server.state
    app.run_analysis()
    assert st.nt_error  # 데이터 없음 보고
    st.nt_error = ""
    app.run_pod()
    assert st.nt_error
    st.nt_error = ""
    app.build_twin()
    assert st.nt_error


def test_build_ui_if_gl_available() -> None:
    """전체 UI(PyVista 뷰어) 빌드 — GL 미지원 환경에서는 skip."""
    from naviertwin.web.app import create_web_app

    try:
        app = create_web_app(build_ui=True, server=_gl_server())
    except Exception as exc:  # noqa: BLE001 — GL/render 컨텍스트 부재
        pytest.skip(f"GL 렌더 컨텍스트 없음: {exc}")
    assert app.plotter is not None
    assert callable(getattr(app.ctrl, "view_update", None))


def test_timestep_change_renders_if_gl() -> None:
    """타임스텝 변경 시 실제 plotter 렌더 경로(_render)를 GL 가능 환경에서 검증."""
    from trame.app import get_server

    from naviertwin.web.app import create_web_app

    try:
        app = create_web_app(build_ui=True, server=get_server("nt-ts-render", client_type="vue3"))
    except Exception as exc:  # noqa: BLE001 — GL 부재
        pytest.skip(f"GL 렌더 컨텍스트 없음: {exc}")
    app.load_demo()
    app.server.state.nt_timestep = 3
    app._on_view_state_change()
    assert app.server.state.nt_error == ""


def _gl_server():
    from trame.app import get_server

    return get_server("nt-test-ui", client_type="vue3")
