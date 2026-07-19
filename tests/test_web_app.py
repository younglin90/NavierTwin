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
        "nt_scale_summary",
        "nt_sensor_summary",
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
        "nt_plan_scale",
        "nt_sensor_push",
        "nt_sensor_align",
    ]:
        assert callable(getattr(app.ctrl, cb, None)), f"missing controller: {cb}"


def test_operation_scale_plan_and_sensor_alignment() -> None:
    app = _make_app("nt-test-operation")
    st = app.server.state

    app.plan_large_workload()
    assert "point chunk" in st.nt_scale_summary
    assert st.nt_error == ""

    st.nt_sensor_id = "p1"
    st.nt_sensor_ids = "p1"
    st.nt_sensor_timestamp = 2.5
    st.nt_sensor_values = "101325.0"
    app.push_sensor_sample()
    app.align_sensor_samples()
    assert st.nt_sensor_summary.startswith("READY")
    assert "101325" in st.nt_sensor_summary


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


def test_train_field_is_independent_of_viewer_field() -> None:
    """②Model 출력 필드 선택은 3D 뷰어 컬러링용 nt_field 와 완전히 독립적이다."""
    app = _make_app("nt-test-train-field")
    st = app.server.state
    app.load_demo()
    assert set(st.nt_train_field_choices) == {"U", "p"}
    assert st.nt_train_field == "p"  # preferred_field 기본값

    # 뷰어 필드를 U 로 바꿔도 학습 대상(nt_train_field)은 그대로 p.
    st.nt_field = "U"
    app.build_twin()
    assert st.nt_error == ""
    assert "field='p'" in st.nt_model_summary

    # 출력 필드를 명시적으로 U 로 바꾸면 그걸로 학습된다 (뷰어와 무관하게).
    st.nt_field = "p"
    st.nt_train_field = "U"
    app.build_twin()
    assert st.nt_error == ""
    assert "field='U'" in st.nt_model_summary


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
    assert "연산자 랩" in st.nt_error


@pytest.mark.asyncio
async def test_managed_training_job_writes_and_restores_checkpoint() -> None:
    from pathlib import Path

    app = _make_app("nt-test-managed-training")
    st = app.server.state
    app.load_demo()
    st.nt_model_method = "rom"
    st.nt_n_modes = 3

    await app._train_job_async()

    assert st.nt_training_state == "completed"
    assert st.nt_training_cancelable is False
    assert st.nt_training_preflight
    assert Path(st.nt_training_checkpoint).exists()
    assert app.engine is not None

    app.engine = None
    st.nt_model_ready = False
    app.restore_training_checkpoint()
    assert st.nt_error == ""
    assert app.engine is not None
    assert st.nt_model_ready is True
    app._training_jobs.shutdown()


def test_physics_multi_output_train_and_predict() -> None:
    """Physics AI 다중 출력 + 벡터 성분 보존 (v5.0): U 방향 채널이 뷰어에 뜬다."""
    app = _make_app("nt-test-multi-output")
    st = app.server.state
    app.load_demo()

    st.nt_model_method = "physics"
    st.nt_train_fields = ["p", "U"]
    st.nt_physics_epochs = 3  # 테스트 속도용
    st.nt_physics_hidden = 8
    app.build_twin()
    assert st.nt_error == ""
    assert st.nt_physics_ready is True
    assert "다중 출력 2개" in st.nt_model_summary

    st.nt_twin_param = 0.5 * (st.nt_twin_min + st.nt_twin_max)
    app.predict()
    assert st.nt_error == ""
    assert "twin_p" in st.nt_fields
    # 방향이 보존된다 — 성분 채널 + 파생 크기 둘 다.
    for name in ("twin_U_x", "twin_U_y", "twin_U_mag"):
        assert name in st.nt_fields, f"{name} 이 뷰어 필드에 없습니다"
    # 다중 출력 파생 필드는 학습 대상 선택지에 새지 않는다.
    assert "twin_p" not in st.nt_train_field_choices
    assert "twin_U_x" not in st.nt_train_field_choices


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
    from naviertwin.web import service

    app = _make_app("nt-test-compare")
    st = app.server.state
    app.load_demo()
    app.run_compare()
    assert st.nt_error == ""
    assert st.nt_compare_dialog is True
    # ROM (reducer × 리더보드 surrogate) 조합 + Physics AI(직접 회귀) 1행.
    # ezyrb_ann 은 느려서 기본 리더보드에서 빠진다 (LEADERBOARD_SURROGATES).
    n_rom = len(service.REDUCERS) * len(service.LEADERBOARD_SURROGATES)
    assert len(st.nt_compare_rows) == n_rom + 1
    assert any("physicsnemo" in r["combo"] for r in st.nt_compare_rows)
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
    original_project_id = app.workspace.project.project_id
    st.nt_export_dir = str(tmp_path)
    app.export_project()
    ntwin = next(f for f in os.listdir(tmp_path) if f.endswith(".ntwin"))
    assert list(tmp_path.glob("*.manifest.json"))

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
    assert app.workspace.project.project_id == original_project_id
    assert st.nt_project_id == original_project_id
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


def _built_html() -> str:
    """실제로 생성된 vue 마크업 (소스 텍스트가 아니라 빌드 결과)."""
    from trame.app import get_server

    from naviertwin.web.app import NavierTwinWebApp

    app = NavierTwinWebApp(server=get_server("nt-test-html", client_type="vue3"))
    try:
        return str(app.build_ui().html)
    except (NameError, TypeError):
        raise
    except Exception as exc:  # noqa: BLE001 — GL/render 컨텍스트 부재
        pytest.skip(f"GL 렌더 컨텍스트 없음: {exc}")


def test_long_explanations_live_in_tooltips_not_panels() -> None:
    """긴 설명은 패널 본문이 아니라 ⓘ 툴팁 안에 있어야 한다.

    캡션을 다시 패널에 눌러 담으면(원래 상태로 되돌리면) 여기서 깨진다.
    """
    html = _built_html()
    for phrase in (
        "탐색기에서 CFD 파일",
        "폴더 하나 = 케이스 여러 개",
        "성긴 균일 격자로 보간 재샘플합니다",
        "신경장(Physics AI)은 좌표를 입력으로 받아",
    ):
        assert phrase in html, f"설명이 사라졌습니다: {phrase}"
        # 설명 앞의 가장 가까운 태그가 VTooltip 이어야 한다 = 툴팁 안에 있다.
        before = html[: html.index(phrase)]
        assert before.rindex("<VTooltip") > before.rindex("<VExpansionPanelText"), (
            f"설명이 툴팁 밖(패널 본문)에 있습니다: {phrase}"
        )


def test_warnings_get_the_orange_icon_and_explanations_the_grey_one() -> None:
    """경고와 설명은 아이콘 색으로 구분된다 — 화면을 비우되 신호는 남긴다."""
    html = _built_html()

    # DMD 부적합 경고: 못 보면 결과가 조용히 틀어지므로 반드시 ⚠ 여야 한다.
    dmd = html[: html.index("상태의 시간 전이 규칙을 학습해")]
    dmd_icon = dmd[dmd.rindex("<VIcon") :]
    assert "mdi-alert-circle-outline" in dmd_icon
    assert 'color="warning"' in dmd_icon

    # 일반 설명은 회색 ⓘ.
    info = html[: html.index("탐색기에서 CFD 파일")]
    info_icon = info[info.rindex("<VIcon") :]
    assert "mdi-information-outline" in info_icon
    assert 'color="grey"' in info_icon

    # 두 종류가 모두 실제로 쓰이고 있다.
    assert html.count("mdi-alert-circle-outline") >= 3
    assert html.count("mdi-information-outline") >= 5


def test_live_status_stays_visible_outside_tooltips() -> None:
    """상태·결과는 툴팁에 숨기면 안 된다 — 호버해야 보이면 기능이 죽는다."""
    html = _built_html()
    for binding in (
        "{{ nt_coarsen_preview }}",  # 해상도 선택의 근거
        "{{ nt_fft_summary }}",  # 분석 결과
        "{{ nt_model_summary }}",  # 학습 결과
        "{{ nt_error }}",  # 오류
    ):
        assert binding in html, f"상태 표시가 사라졌습니다: {binding}"
        before = html[: html.index(binding)]
        last_tooltip = before.rfind("<VTooltip")
        if last_tooltip != -1:
            # 툴팁이 앞에 있더라도 이미 닫힌 뒤여야 한다.
            assert before.rfind("</VTooltip>") > last_tooltip, (
                f"상태가 툴팁 안에 갇혔습니다: {binding}"
            )


def test_split_view_toggle_and_field_selection() -> None:
    """분할 뷰(v5.4): 토글 상태 + 좌(실제)/우(트윈) 필드 선택 로직."""
    app = _make_app("nt-test-split-view")
    st = app.server.state
    st.nt_demo_kind = "sweep"
    app.load_demo()
    st.nt_model_method = "rom"
    app.build_twin()
    assert st.nt_error == ""

    # 예측 전: 분할 필드가 없으므로 (빈, 빈).
    left, right = app._split_field_names()
    assert left and not right  # 실제 필드는 있지만 트윈 필드는 아직 없음

    st.nt_twin_params = [0.5 * (a + b) for a, b in zip(st.nt_twin_mins, st.nt_twin_maxs)]
    app.predict()
    assert st.nt_error == ""

    left, right = app._split_field_names()
    assert left == st.nt_train_field
    assert right.startswith("twin_")

    # 토글: 켜면 nt_split_view True, 다시 누르면 False.
    assert st.nt_split_view is False
    app.toggle_split_view()
    assert st.nt_split_view is True
    app.toggle_split_view()
    assert st.nt_split_view is False


def test_split_view_syncs_timestep_to_twin_prediction() -> None:
    """분할 뷰 켜짐 상태에서 타임스텝을 바꾸면 시간 파라미터 트윈(DMD)의
    우측 예측이 그 시간으로 자동 재계산된다 (검토 §6½ #9 잔여분).

    ``nt_timestep`` 은 좌측(실제) 뷰어의 시간 인덱스, ``nt_twin_param`` 은
    트윈(DMD)의 시간 입력이다 — 분할 뷰가 켜져 있으면 전자가 후자를
    따라가야 한다.
    """
    import numpy as np

    app = _make_app("nt-test-split-timestep-sync")
    st = app.server.state
    app.load_demo()

    st.nt_model_method = "dynamics"
    app.build_twin()
    assert st.nt_error == ""
    assert st.nt_dmd_ready is True

    st.nt_twin_param = 0.0
    app.predict()
    assert st.nt_error == ""
    assert st.nt_predicted is True
    before = np.asarray(app.dataset.mesh.point_data["twin_prediction"]).copy()

    app.toggle_split_view()
    assert st.nt_split_view is True

    # 좌측 타임스텝을 학습 구간 안의 다른 지점으로 옮긴다 — trame 테스트
    # 하네스는 이벤트 루프가 없어 state.change 콜백이 자동 발화하지 않으므로
    # (기존 test_timestep_change_renders_if_gl 과 동일한 패턴) 직접 호출한다.
    new_idx = max(1, int(st.nt_nsteps) - 1)
    st.nt_timestep = new_idx
    app._on_split_timestep_sync()
    assert st.nt_error == ""
    assert st.nt_twin_param == pytest.approx(float(app.dataset.time_steps[new_idx]))

    after = np.asarray(app.dataset.mesh.point_data["twin_prediction"])
    assert not np.allclose(before, after), (
        "타임스텝 변경이 분할 뷰 우측(트윈) 예측에 반영되지 않았습니다"
    )

    # 분할 뷰를 끄면 자동 동기화도 멈춘다 — nt_twin_param 을 더는 건드리지 않는다.
    app.toggle_split_view()
    assert st.nt_split_view is False
    stale_param = st.nt_twin_param
    st.nt_timestep = 0
    app._on_split_timestep_sync()
    assert st.nt_twin_param == pytest.approx(stale_param), (
        "분할 뷰가 꺼져 있으면 시간 자동 동기화가 없어야 합니다"
    )


def test_split_view_timestep_sync_is_noop_for_steady_case_set() -> None:
    """정상(steady) 파라미터 스윕 트윈은 시간 파라미터가 없다 — 조용한 no-op.

    ``nt_case_mode=True`` 이고 ``nt_param_names`` 에 ``"t"`` 가 없으면
    (케이스당 스냅샷 1개, 운전조건만 학습) 타임스텝 변경이 트윈 예측을
    건드리면 안 된다 — 애초에 그 트윈은 시간을 입력받지 않는다.
    """
    app = _make_app("nt-test-split-timestep-sync-steady")
    st = app.server.state
    st.nt_demo_kind = "sweep"
    app.load_demo()
    st.nt_model_method = "rom"
    app.build_twin()
    assert st.nt_error == ""
    assert "t" not in list(st.nt_param_names)

    st.nt_twin_params = [0.5 * (a + b) for a, b in zip(st.nt_twin_mins, st.nt_twin_maxs)]
    app.predict()
    assert st.nt_error == ""
    assert st.nt_predicted is True

    app.toggle_split_view()
    assert st.nt_split_view is True

    stale = list(st.nt_twin_params)
    st.nt_timestep = 1
    app._on_split_timestep_sync()  # no-op 이어야 한다 — 에러도, 값 변경도 없음
    assert st.nt_error == ""
    assert list(st.nt_twin_params) == stale


def test_build_ui_if_gl_available() -> None:
    """전체 UI(PyVista 뷰어) 빌드 — GL 미지원 환경에서만 skip.

    이 테스트는 UI 코드 전체를 실제로 빌드하는 유일한 곳이다. 예전엔 모든
    예외를 skip 으로 삼켜서 UI 코드의 NameError 조차 "GL 없음"으로 위장됐다
    (실제로 그렇게 버그를 놓쳤다). 명백한 프로그래밍 오류는 다시 던진다.
    """
    from naviertwin.web.app import create_web_app

    try:
        app = create_web_app(build_ui=True, server=_gl_server())
    except (NameError, TypeError, SyntaxError, IndentationError):
        raise  # UI 코드 버그 — skip 으로 덮으면 안 된다
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


def _make_case_set(directory, *, n_cases=4):
    """정상해 케이스 N개(.vtk) + 파라미터 CSV 폴더를 만든다 (app 레벨 테스트용)."""
    import numpy as np
    import pyvista as pv

    directory.mkdir(parents=True, exist_ok=True)
    velocities = [1.0 + i for i in range(n_cases)]
    for index, velocity in enumerate(velocities):
        grid = pv.ImageData(dimensions=(5, 5, 1))
        coords = np.asarray(grid.points, dtype=float)
        grid.point_data["p"] = velocity * (coords[:, 0] + 2.0 * coords[:, 1])
        grid.save(directory / f"case_{index:02d}.vtk")
    rows = ["inlet_velocity"] + [f"{v}" for v in velocities]
    (directory / "params.csv").write_text("\n".join(rows) + "\n")
    return velocities


def test_case_set_load_train_predict(tmp_path) -> None:
    """케이스 세트(문제 유형 B): 로드 → 운전조건 ROM 학습 → 파라미터 예측."""
    _make_case_set(tmp_path / "sweep")
    app = _make_app("nt-test-caseset")
    st = app.server.state

    st.nt_path = str(tmp_path / "sweep")
    app.load_case_set()
    assert st.nt_error == ""
    assert st.nt_case_mode is True
    assert st.nt_case_count == 4
    assert st.nt_param_names == ["inlet_velocity"]
    assert "케이스 세트" in st.nt_method_hint
    # 파라미터별 슬라이더 상태가 준비된다 (파라미터 1개 → 배열 길이 1).
    assert st.nt_twin_mins == [1.0]
    assert st.nt_twin_maxs == [4.0]
    assert len(st.nt_twin_params) == 1

    app.build_twin()
    assert st.nt_error == ""
    assert st.nt_model_ready is True
    assert "케이스 4개" in st.nt_model_summary
    assert "inlet_velocity" in st.nt_model_summary

    st.nt_twin_params = [2.0]
    app.predict()
    assert st.nt_error == ""
    assert "twin_prediction" in st.nt_fields
    assert "inlet_velocity=2" in st.nt_status

    # 케이스 세트에서는 자동 비교(리더보드)가 명확한 안내와 함께 거부된다.
    app.run_compare()
    assert "케이스 세트" in st.nt_error


def test_case_set_physics_ai_multi_param(tmp_path) -> None:
    """케이스 세트 + Physics AI: 입력이 (좌표 + 운전조건) 으로 확장된다."""
    _make_case_set(tmp_path / "sweep")
    app = _make_app("nt-test-caseset-physics")
    st = app.server.state

    st.nt_path = str(tmp_path / "sweep")
    app.load_case_set()
    st.nt_model_method = "physics"
    st.nt_physics_epochs = 3
    st.nt_physics_hidden = 8
    app.build_twin()
    assert st.nt_error == ""
    assert st.nt_physics_ready is True
    assert "inlet_velocity" in st.nt_model_summary

    st.nt_twin_params = [2.5]
    app.predict()
    assert st.nt_error == ""
    assert "twin_prediction" in st.nt_fields


def test_single_load_clears_case_mode(tmp_path) -> None:
    """케이스 세트 뒤 단일 데이터셋을 로드하면 케이스 상태가 완전히 리셋된다."""
    _make_case_set(tmp_path / "sweep")
    app = _make_app("nt-test-caseset-reset")
    st = app.server.state

    st.nt_path = str(tmp_path / "sweep")
    app.load_case_set()
    assert st.nt_case_mode is True

    app.load_demo()
    assert st.nt_case_mode is False
    assert st.nt_case_count == 0
    assert st.nt_param_names == []
    assert app.case_datasets is None


def test_predict_on_other_mesh_and_restore(tmp_path) -> None:
    """M3: 다른 격자에 예측 → 뷰어 전환(학습 상태 보존) → 학습 격자 복귀."""
    import pyvista as pv

    _make_case_set(tmp_path / "sweep")
    app = _make_app("nt-test-predict-mesh")
    st = app.server.state

    st.nt_path = str(tmp_path / "sweep")
    app.load_case_set()
    st.nt_model_method = "physics"
    st.nt_physics_epochs = 3
    st.nt_physics_hidden = 8
    app.build_twin()
    assert st.nt_physics_ready is True
    origin_points = st.nt_info_points
    engine = app.engine

    # 학습 격자(5×5=25)와 다른 해상도(7×7=49)의 대상 메쉬.
    fine = pv.ImageData(dimensions=(7, 7, 1), spacing=(4 / 6, 4 / 6, 1))
    fine.save(tmp_path / "target.vtk")

    st.nt_twin_params = [2.0]
    st.nt_path = str(tmp_path / "target.vtk")
    app.predict_on_mesh()
    assert st.nt_error == ""
    assert st.nt_predict_mesh_name == "target.vtk"
    assert st.nt_info_points == 49  # 뷰어가 대상 격자로 전환됨
    assert "twin_p" in st.nt_fields
    # 학습 상태는 그대로 보존된다 (엔진/케이스 모드).
    assert app.engine is engine
    assert st.nt_case_mode is True
    assert st.nt_model_ready is True

    # 예측 격자 모드에서도 케이스 세트는 case_datasets 로 재학습 가능하다.
    app.build_twin()
    assert st.nt_error == ""

    app.restore_training_mesh()
    assert st.nt_predict_mesh_name == ""
    assert st.nt_info_points == origin_points
    assert app.engine is not None


def test_predict_on_mesh_rejects_rom_engine(tmp_path) -> None:
    """ROM 트윈은 학습 메쉬에 묶여 다른 격자 예측이 거부된다 (명확한 안내)."""
    import pyvista as pv

    app = _make_app("nt-test-predict-mesh-rom")
    st = app.server.state
    app.load_demo()
    app.build_twin()  # 기본 ROM
    assert st.nt_model_ready is True

    fine = pv.ImageData(dimensions=(5, 5, 1))
    fine.save(tmp_path / "target.vtk")
    st.nt_path = str(tmp_path / "target.vtk")
    app.predict_on_mesh()
    assert "Physics AI" in st.nt_error


def test_train_guarded_while_on_predict_mesh(tmp_path) -> None:
    """시계열 트윈에서 예측 격자로 전환한 상태면 재학습을 막는다."""
    import pyvista as pv

    app = _make_app("nt-test-predict-mesh-guard")
    st = app.server.state
    app.load_demo()
    st.nt_model_method = "physics"
    st.nt_physics_epochs = 3
    st.nt_physics_hidden = 8
    app.build_twin()

    fine = pv.ImageData(dimensions=(5, 5, 1), spacing=(6.3 / 4, 6.3 / 4, 1))
    fine.save(tmp_path / "target.vtk")
    st.nt_twin_param = 1.0
    st.nt_path = str(tmp_path / "target.vtk")
    app.predict_on_mesh()
    assert st.nt_error == ""
    assert st.nt_predict_mesh_name == "target.vtk"

    app.build_twin()
    assert "학습 격자로 복귀" in st.nt_error


def test_dmd_dynamics_method_forecasts_beyond_training() -> None:
    """계열 Ⓓ: DMD 선택 시 슬라이더가 학습 구간 밖까지 열리고 적합도를 보고한다."""
    app = _make_app("nt-test-dmd")
    st = app.server.state
    app.load_demo()

    st.nt_model_method = "dynamics"
    app.build_twin()
    assert st.nt_error == ""
    assert st.nt_dmd_ready is True
    assert st.nt_physics_ready is False
    assert "DMD" in st.nt_model_summary
    # 외삽 허용: 슬라이더 상한 > 학습 상한.
    assert st.nt_twin_max > st.nt_twin_train_max
    # 적합도가 보고된다 (필라멘트 데모는 DMD 부적합 → 큰 오차가 나와야 정상).
    assert st.nt_dmd_fit_error > 0.0
    assert "재구성 오차" in st.nt_status

    # 학습 구간 밖에서 예측이 실제로 동작한다.
    st.nt_twin_param = st.nt_twin_max
    app.predict()
    assert st.nt_error == ""
    assert "twin_prediction" in st.nt_fields

    # 다른 방식으로 재학습하면 DMD 상태가 꺼진다.
    st.nt_model_method = "rom"
    app.build_twin()
    assert st.nt_dmd_ready is False
    assert st.nt_twin_max == st.nt_twin_train_max  # 내삽 전용으로 복귀


def test_dmd_rejected_for_steady_case_sets(tmp_path) -> None:
    """정상(스텝 1개) 케이스 세트는 시간 전개가 없어 동역학 예보가 거부된다.

    v5.2 부터 **비정상** 케이스 세트는 ParametricDMD 로 가능하다 — 거절은
    "케이스 세트여서"가 아니라 "시간축이 없어서"다.
    """
    _make_case_set(tmp_path / "sweep")
    app = _make_app("nt-test-dmd-caseset")
    st = app.server.state
    st.nt_path = str(tmp_path / "sweep")
    app.load_case_set()

    st.nt_model_method = "dynamics"
    app.build_twin()
    assert "타임스텝 4개 이상" in st.nt_error


def test_unsteady_sweep_demo_end_to_end() -> None:
    """비정상 스윕 데모(v5.0): 케이스+타임스텝 슬라이더 공존, (μ, t) 트윈 학습.

    예전에는 케이스 세트가 마지막 스텝만 남겨 시간축이 사라졌다 — 이 테스트가
    그 붕괴의 회귀 방지다.
    """
    app = _make_app("nt-test-unsteady-sweep")
    st = app.server.state
    st.nt_demo_kind = "sweep_unsteady"
    app.load_demo()
    assert st.nt_error == ""
    # 케이스 슬라이더와 타임스텝 슬라이더가 **둘 다** 살아 있다.
    assert st.nt_case_mode is True
    assert st.nt_case_count == 4
    assert st.nt_has_timesteps is True
    assert st.nt_nsteps == 8
    # 전략 판정: ROM/Physics/동역학(ParametricDMD, v5.2) 전부 가능.
    status = st.nt_strategy_status
    assert status["rom"]["ok"] and status["physics"]["ok"]
    assert status["dynamics"]["ok"]
    assert "시간(t)도 입력 파라미터" in st.nt_method_hint

    # ROM 학습 → 슬라이더가 μ 와 t 두 개.
    st.nt_model_method = "rom"
    app.build_twin()
    assert st.nt_error == ""
    assert st.nt_param_names == ["inlet_velocity", "t"]
    assert len(st.nt_twin_params) == 2
    assert st.nt_twin_maxs[1] == pytest.approx(1.4)

    # (μ, t) 예측이 t 에 실제로 반응한다.
    import numpy as np

    st.nt_twin_params = [15.0, 0.2]
    app.predict()
    assert st.nt_error == ""
    early = np.asarray(app.dataset.mesh.point_data["twin_prediction"]).copy()
    st.nt_twin_params = [15.0, 1.2]
    app.predict()
    late = np.asarray(app.dataset.mesh.point_data["twin_prediction"])
    assert not np.allclose(early, late), "t 슬라이더가 예측에 반영되지 않습니다"

    # 동역학(v5.2): ParametricDMD 가 전체 케이스로 학습되고, t 슬라이더 상한이
    # 학습 구간(1.4)을 넘어 예보 구간까지 열린다 — 이 계열의 존재 이유.
    st.nt_model_method = "dynamics"
    app.build_twin()
    assert st.nt_error == ""
    assert st.nt_dmd_ready is True
    assert "ParametricDMD" in st.nt_model_summary
    assert st.nt_param_names == ["inlet_velocity", "t"]
    assert st.nt_twin_maxs[1] > 1.4 + 1e-9, "t 예보 상한이 학습 구간을 못 넘습니다"

    st.nt_twin_params = [17.5, 1.9]  # 학습에 없던 μ + 학습 구간 밖 t
    app.predict()
    assert st.nt_error == ""
    forecast = np.asarray(app.dataset.mesh.point_data["twin_prediction"])
    assert np.isfinite(forecast).all()


def test_support_status_ood_levels_in_app() -> None:
    """③Twin OOD 3단계(v5.6): 학습 범위 내부/밖 예측이 상태에 반영된다."""
    app = _make_app("nt-test-support")
    st = app.server.state
    st.nt_demo_kind = "sweep"
    app.load_demo()
    st.nt_model_method = "rom"
    app.build_twin()
    assert st.nt_error == ""

    # 학습 범위 중앙 예측 → IN_SUPPORT.
    lo = list(st.nt_twin_mins)
    hi = list(st.nt_twin_maxs)
    st.nt_twin_params = [0.5 * (a + b) for a, b in zip(lo, hi)]
    app.predict()
    assert st.nt_error == ""
    assert st.nt_support_status == "IN_SUPPORT"

    # 범위를 크게 벗어난 예측 → OUT_OF_SUPPORT.
    st.nt_twin_params = [b + (b - a) for a, b in zip(lo, hi)]
    app.predict()
    assert st.nt_error == ""
    assert st.nt_support_status == "OUT_OF_SUPPORT"
    assert st.nt_support_label  # 한국어 라벨 존재


def test_device_badge_reflects_training_backend() -> None:
    """학습 디바이스 배지(v5.6): ROM=CPU, 신경망=GPU/CPU(가용성 따라).

    GPU 로 학습되는데 사용자가 모르는 문제를 해소한다. device_used 는 A3 가
    모델 metadata 에 기록한 값을 읽는다.
    """
    import torch

    app = _make_app("nt-test-device-badge")
    st = app.server.state
    app.load_demo()

    # ROM 은 CPU 전용 — 배지가 CPU.
    st.nt_model_method = "rom"
    app.build_twin()
    assert st.nt_error == ""
    assert st.nt_train_device == "CPU"

    # 신경망(Physics AI)은 실제 학습 디바이스를 반영한다.
    st.nt_model_method = "physics"
    st.nt_physics_epochs = 3
    app.build_twin()
    assert st.nt_error == ""
    expected = "GPU" if torch.cuda.is_available() else "CPU"
    assert st.nt_train_device.startswith(expected), st.nt_train_device


def test_strategy_status_follows_loaded_data() -> None:
    """전략 카드 상태가 데이터를 따라간다 — 시계열↔단일스냅샷 전환."""
    app = _make_app("nt-test-strategy-status")
    st = app.server.state
    app.load_demo()  # filament 시계열
    assert st.nt_strategy_status["dynamics"]["ok"] is True
    assert st.nt_strategy_status["operator"]["ok"] is False

    st.nt_demo_kind = "sweep"
    app.load_demo()
    assert st.nt_strategy_status["rom"]["ok"] is True
    assert st.nt_strategy_status["dynamics"]["ok"] is False


def test_coarsen_reduces_dataset_and_resets_model() -> None:
    """①Import 해상도 낮추기: 점 수가 줄고 파생 상태가 리셋된다."""
    app = _make_app("nt-test-coarsen")
    st = app.server.state
    app.load_demo()
    before = st.nt_info_points
    app.build_twin()
    assert st.nt_model_ready is True

    st.nt_coarsen_resolution = 16
    app.coarsen_current()
    assert st.nt_error == ""
    assert st.nt_info_points < before
    assert "→" in st.nt_coarsen_summary
    # 성긴 데이터셋은 새 데이터셋이므로 모델 상태가 리셋된다.
    assert st.nt_model_ready is False
    assert app.engine is None
    # 시계열과 필드는 살아있어 바로 재학습 가능하다.
    assert st.nt_has_timesteps is True
    app.build_twin()
    assert st.nt_error == ""
    assert st.nt_model_ready is True


def test_coarsen_preview_tracks_slider_and_predicts_the_result() -> None:
    """해상도를 고르는 동안 결과 크기가 미리 보이고, 그 예고가 실제와 맞는다."""
    from naviertwin.web import service

    app = _make_app("nt-test-coarsen-preview")
    st = app.server.state
    # 데이터가 없으면 미리 볼 것도 없다.
    assert st.nt_coarsen_preview == ""

    app.load_demo()
    assert st.nt_coarsen_preview != ""  # 로드 직후 기본 해상도로 예고

    # 슬라이더를 움직이면 미리보기가 따라온다.
    st.nt_coarsen_resolution = 16
    app._update_coarsen_preview()
    coarse_preview = st.nt_coarsen_preview
    st.nt_coarsen_resolution = 64
    app._update_coarsen_preview()
    assert st.nt_coarsen_preview != coarse_preview

    # 예고한 점 수가 실제 적용 결과와 같아야 미리보기에 의미가 있다.
    st.nt_coarsen_resolution = 20
    app._update_coarsen_preview()
    predicted = service.estimate_coarsen(app.dataset, 20)["points_after"]
    app.coarsen_current()
    assert st.nt_error == ""
    assert int(st.nt_info_points) == predicted


def test_coarsen_preview_warns_when_resolution_would_add_points() -> None:
    """해상도가 원본보다 촘촘하면 '낮추기'가 아니므로 경고로 알린다.

    데모(48×48)에 기본값 48을 그대로 쓰면 점이 되레 늘어난다 — 사용자가 이걸
    모르고 적용하면 메모리를 아끼려다 늘리게 된다.
    """
    app = _make_app("nt-test-coarsen-warn")
    st = app.server.state
    app.load_demo()

    st.nt_coarsen_resolution = 8  # 확실히 성기게
    app._update_coarsen_preview()
    assert st.nt_coarsen_increases is False
    assert "축소" in st.nt_coarsen_preview

    st.nt_coarsen_resolution = 128  # 원본보다 촘촘하게
    app._update_coarsen_preview()
    assert st.nt_coarsen_increases is True
    assert "증가" in st.nt_coarsen_preview
    assert "축소" not in st.nt_coarsen_preview


def test_coarsen_preview_hidden_for_case_sets(tmp_path) -> None:
    """케이스 세트는 로드 시 이미 재샘플되므로 미리보기가 뜨면 안 된다."""
    _make_case_set(tmp_path / "sweep")
    app = _make_app("nt-test-coarsen-preview-caseset")
    st = app.server.state
    st.nt_path = str(tmp_path / "sweep")
    app.load_case_set()
    assert st.nt_coarsen_preview == ""


def test_case_set_resolution_is_user_selectable(tmp_path) -> None:
    """형상 가변 케이스 세트의 공통 격자 해상도를 사용자가 정할 수 있다."""
    app = _make_app("nt-test-case-resolution")
    st = app.server.state
    # 형상이 다른 데모(shapes)는 메쉬가 달라 공통 격자로 재샘플된다.
    st.nt_demo_kind = "shapes"
    st.nt_case_resolution = 12
    app.load_demo()
    assert st.nt_error == ""
    coarse_points = int(st.nt_info_points)
    assert st.nt_case_resampled is True

    st.nt_case_resolution = 28
    app.load_demo()
    assert st.nt_error == ""
    # 해상도를 올리면 실제로 격자가 촘촘해진다 — 값이 배선돼 있다는 증거.
    assert int(st.nt_info_points) > coarse_points
    assert "28" in st.nt_case_grid_summary or "29" in st.nt_case_grid_summary


def test_coarsen_rejected_for_case_sets(tmp_path) -> None:
    _make_case_set(tmp_path / "sweep")
    app = _make_app("nt-test-coarsen-caseset")
    st = app.server.state
    st.nt_path = str(tmp_path / "sweep")
    app.load_case_set()
    app.coarsen_current()
    assert "케이스 세트" in st.nt_error


def test_input_fields_wired_to_physics_training() -> None:
    """②Model 입력 필드 선택이 학습에 반영되고 요약에 표시된다."""
    app = _make_app("nt-test-input-fields")
    st = app.server.state
    app.load_demo()

    st.nt_model_method = "physics"
    st.nt_train_field = "p"
    st.nt_train_fields = ["p"]
    st.nt_train_input_fields = ["U"]
    st.nt_physics_epochs = 3
    st.nt_physics_hidden = 8
    app.build_twin()
    assert st.nt_error == ""
    assert "입력 U+시간(t)" in st.nt_model_summary
    assert app.engine.model.input_field_names == ["U"]

    # 예측 슬라이더는 그대로 동작한다 (학습 입력장을 시간 보간해 채움).
    st.nt_twin_param = 0.5 * (st.nt_twin_min + st.nt_twin_max)
    app.predict()
    assert st.nt_error == ""
    assert "twin_prediction" in st.nt_fields


def test_input_field_equal_to_output_is_filtered() -> None:
    """출력으로 고른 필드를 입력에도 넣으면 조용히 제외된다 (항등 학습 방지)."""
    app = _make_app("nt-test-input-overlap")
    st = app.server.state
    app.load_demo()
    st.nt_model_method = "physics"
    st.nt_train_fields = ["p"]
    st.nt_train_input_fields = ["p", "U"]  # p 는 출력이므로 제외되어야 함
    assert app._training_input_fields() == ["U"]

    st.nt_train_input_fields = ["p"]  # 전부 제외 → 입력 없음
    assert app._training_input_fields() == []
    st.nt_physics_epochs = 3
    st.nt_physics_hidden = 8
    app.build_twin()
    assert st.nt_error == ""
    assert "입력 시간(t)" in st.nt_model_summary


def test_demo_catalog_loads_time_series_and_case_sets() -> None:
    """데모 선택이 시계열/케이스 세트 경로로 각각 올바르게 디스패치된다."""
    app = _make_app("nt-test-demo-catalog")
    st = app.server.state
    assert st.nt_demo_kind == "filament"
    assert len(st.nt_demo_choices) == len(app_service().DEMO_CATALOG)

    # 시계열 데모
    st.nt_demo_kind = "waves"
    app.load_demo()
    assert st.nt_error == ""
    assert st.nt_case_mode is False
    assert st.nt_has_timesteps is True
    assert "진행파" in st.nt_status

    # 케이스 세트 데모 — 같은 버튼이 케이스 세트 경로로 간다.
    st.nt_demo_kind = "sweep"
    app.load_demo()
    assert st.nt_error == ""
    assert st.nt_case_mode is True
    assert st.nt_case_count == 5
    assert st.nt_param_names == ["inlet_velocity", "angle_of_attack"]

    # 형상 가변 데모 — 자동 재샘플 안내가 뜬다.
    st.nt_demo_kind = "shapes"
    app.load_demo()
    assert st.nt_error == ""
    assert st.nt_case_resampled is True
    assert st.nt_param_names == ["radius"]
    assert "형상 가변" in st.nt_method_hint

    # 다시 시계열 데모로 돌아오면 케이스 상태가 리셋된다.
    st.nt_demo_kind = "filament"
    app.load_demo()
    assert st.nt_case_mode is False
    assert st.nt_case_count == 0


def app_service():
    from naviertwin.web import service

    return service


def test_waves_demo_makes_dmd_usable() -> None:
    """waves 데모가 있어야 동역학 예보(DMD)를 실제로 시험할 수 있다."""
    app = _make_app("nt-test-demo-waves-dmd")
    st = app.server.state
    st.nt_demo_kind = "waves"
    app.load_demo()

    st.nt_model_method = "dynamics"
    app.build_twin()
    assert st.nt_error == ""
    assert st.nt_dmd_ready is True
    # 필라멘트(0.66)와 달리 적합도가 초록 신호등(<10%)에 들어와야 한다.
    assert st.nt_dmd_fit_error < 0.1

    st.nt_twin_param = st.nt_twin_max  # 학습 구간 밖 외삽
    app.predict()
    assert st.nt_error == ""
    assert "twin_prediction" in st.nt_fields


def test_case_selector_shows_each_case_and_preserves_model() -> None:
    """케이스 세트: 슬라이더로 케이스별 원본 해를 보되 학습 상태는 유지된다."""
    import numpy as np

    app = _make_app("nt-test-case-selector")
    st = app.server.state
    st.nt_demo_kind = "sweep"
    app.load_demo()
    assert st.nt_case_mode is True
    assert st.nt_case_index == 0
    # 케이스별 운전조건 문구가 준비된다.
    assert len(st.nt_case_labels) == st.nt_case_count
    assert "inlet_velocity=10" in st.nt_case_labels[0]

    app.build_twin()
    engine = app.engine
    assert st.nt_model_ready is True

    first = np.asarray(app.dataset.mesh.point_data["p"]).copy()

    # 다른 케이스로 전환 → 실제로 다른 장이 보여야 한다.
    st.nt_case_index = 4
    app.select_case()
    assert st.nt_error == ""
    assert "케이스 5/5" in st.nt_status
    assert "inlet_velocity=30" in st.nt_status
    last = np.asarray(app.dataset.mesh.point_data["p"])
    assert not np.allclose(first, last), "케이스를 바꿔도 장이 같다"

    # 뷰어만 바뀌고 학습 상태(엔진/케이스 모드)는 보존된다.
    assert app.engine is engine
    assert st.nt_case_mode is True
    assert st.nt_model_ready is True
    assert st.nt_case_count == 5

    # 범위를 벗어난 인덱스는 클램프된다.
    st.nt_case_index = 99
    app.select_case()
    assert st.nt_error == ""
    assert "케이스 5/5" in st.nt_status


def test_case_selector_noop_without_case_set() -> None:
    """시계열 데이터에서는 케이스 선택이 아무 일도 하지 않는다."""
    app = _make_app("nt-test-case-selector-noop")
    st = app.server.state
    app.load_demo()
    before = st.nt_status
    app.select_case()
    assert st.nt_status == before
    assert st.nt_error == ""
