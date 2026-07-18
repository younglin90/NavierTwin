"""NavierTwin 웹 GUI — trame 기반 브라우저 워크플로우.

데스크톱 PySide6 GUI 의 핵심 6단계 중 MVP 4단계(Import → Analyze → Reduce →
Twin)를 브라우저에서 제공한다. Kitware ``trame`` + ``pyvista.trame`` 을 사용해
데스크톱 ``VtkViewer`` 와 동일한 PyVista 렌더 파이프라인을 서버사이드로 스트리밍
한다. ``core`` 모듈은 그대로 재사용하며 로직은 :mod:`naviertwin.web.service` 에
Qt/GL 비의존으로 분리되어 있다.

Usage:
    from naviertwin.web.app import create_web_app
    app = create_web_app()
    app.server.start()           # 또는 naviertwin.web.app.run_web()

설계:
    - 로컬 단일 사용자 가정. 하나의 ``trame`` server + 하나의 PyVista Plotter.
    - 워크플로우 로직은 service 계층 호출 → 결과 field 를 3D 뷰어에 표시.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from naviertwin import __version__
from naviertwin.web import bench, render, service, theme

log = logging.getLogger(__name__)

_ANALYSIS_CHOICES = [
    {"title": "Q-criterion", "value": "q_criterion"},
    {"title": "λ₂ Criterion", "value": "lambda2"},
]


def _dark_chart(fn: Any) -> Any:
    """차트 생성 메서드를 다크 matplotlib 스타일 컨텍스트로 감싼다."""

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        with theme.mpl_dark():
            return fn(self, *args, **kwargs)

    wrapper.__name__ = getattr(fn, "__name__", "wrapper")
    return wrapper


class NavierTwinWebApp:
    """trame 기반 NavierTwin 웹 애플리케이션.

    Attributes:
        server: trame server 인스턴스.
        state: trame 공유 state.
        ctrl: trame controller.
        plotter: PyVista Plotter (off_screen, 서버사이드 렌더).
        dataset: 현재 로드된 ``CFDDataset``.
        engine: 학습된 ``TwinEngine`` (Twin 단계).
    """

    def __init__(self, server: Any = None) -> None:
        from trame.app import get_server

        self.server = server if server is not None else get_server(client_type="vue3")
        self.state = self.server.state
        self.ctrl = self.server.controller
        self.plotter: Optional[Any] = None
        self.dataset: Optional[Any] = None
        # 케이스 세트(문제 유형 B) 로드 시에만 채워진다 — dataset 은 뷰어용
        # 첫 케이스를 가리키고, 학습은 case_datasets 전체 + case_params 로 한다.
        self.case_datasets: Optional[list[Any]] = None
        self.case_params: Optional[Any] = None
        self.case_param_names: list[str] = []
        # 예측 격자 전환(M3) 중 원래 학습 데이터셋을 보관 — 복귀용.
        self._origin_dataset: Optional[Any] = None
        self.reducer: Optional[Any] = None
        self.engine: Optional[Any] = None
        self._pod_result: Optional[dict[str, Any]] = None
        self._bench_dataset: Optional[dict[str, Any]] = None
        self._bench_result: Optional[dict[str, Any]] = None
        self._ui_built = False
        # 비동기(executor) 실행 중에는 렌더를 메인 스레드로 미룬다 (GL 안전).
        self._executor: Optional[Any] = None
        self._defer_render = False
        self._render_pending = False
        self._reset_camera_pending = False
        # 라이브 진행 스트리밍 — 워커가 큐에 상태를 넣으면 loop 스레드 모니터가 적용.
        import queue as _queue

        self._progress_queue: Any = _queue.Queue()
        self._progress_monitor: Optional[Any] = None

        self._init_state()
        self._register_callbacks()

    # ------------------------------------------------------------------
    # State / callbacks
    # ------------------------------------------------------------------

    def _init_state(self) -> None:
        st = self.state
        st.nt_version = __version__
        st.nt_status = "데모 데이터를 로드하거나 '경로에서 로드'로 파일을 선택하세요."
        st.nt_busy = False
        st.nt_error = ""
        # 진행률: -1 = 비결정형(스피너), 0~100 = 결정형(라이브 학습/비교)
        st.nt_progress = -1.0
        # 드로어 확장 패널 열림 인덱스 (칩 클릭 → 해당 패널)
        st.nt_open_panels = [0]
        # 토스트 알림
        st.nt_toast_show = False
        st.nt_toast = ""
        st.nt_toast_color = "success"
        st.nt_toast_icon = "mdi-check-circle"

        # Import — 데모 카탈로그 (모델 계열별로 잘 맞는 데이터를 하나씩)
        st.nt_demo_kind = "filament"
        st.nt_demo_choices = [
            {"title": d["title"], "value": d["value"]} for d in service.DEMO_CATALOG
        ]
        st.nt_demo_notes = {d["value"]: d["note"] for d in service.DEMO_CATALOG}

        st.nt_path = ""
        # 파일 브라우저 모달 — "single"(파일/폴더 1개) | "caseset"(폴더=케이스 세트)
        st.nt_fb_mode = "single"
        st.nt_fb_dialog = False
        st.nt_fb_cwd = ""
        st.nt_fb_parent = None
        st.nt_fb_entries = []
        st.nt_fb_home = ""
        st.nt_has_dataset = False
        st.nt_info_points = 0
        st.nt_info_cells = 0
        st.nt_info_steps = 0
        st.nt_info_fields = ""
        st.nt_info_source = ""

        # 케이스 세트 (문제 유형 B — 정상 파라미터 스윕): 파일 N개 = 운전조건
        # N개 + 파라미터 CSV. False 면 문제 유형 A (단일 케이스 시계열, 입력=t).
        # 근거·로드맵: .omc/plans/model-taxonomy-plan.md §13, M2.
        st.nt_case_mode = False
        st.nt_case_count = 0
        st.nt_case_names = []
        st.nt_params_source = ""
        st.nt_param_names = []
        # 케이스 세트에는 시간축이 없으므로 타임스텝 슬라이더 대신 케이스
        # 슬라이더로 케이스별 결과를 본다 (뷰어만 교체 — 학습 상태 보존).
        st.nt_case_index = 0
        st.nt_case_labels = []
        # 형상 가변(M4a): 케이스 메쉬가 서로 다르면 공통 격자로 재샘플했다는 표시.
        st.nt_case_resampled = False
        st.nt_case_grid_summary = ""
        # 해상도 낮추기(coarsen) — 대용량 메쉬 대응. extract_field_snapshots 는
        # (n_features × n_steps) 전체 행렬을 메모리에 올리므로 학습 전에 해상도를
        # 줄이는 것이 유일하게 효과적이다.
        st.nt_coarsen_resolution = 48
        st.nt_coarsen_summary = ""
        # 적용 전 미리보기 — 사용자가 해상도를 고르려면 "이 값이면 몇 점"인지
        # 먼저 보여야 한다. 슬라이더를 움직일 때마다 갱신된다.
        st.nt_coarsen_preview = ""
        st.nt_coarsen_increases = False
        # 케이스 세트의 공통 격자 해상도 — 메쉬가 서로 다를 때(형상 가변) 로드
        # 시점에 재샘플되므로, 로드 전에 정해야 한다.
        st.nt_case_resolution = 32

        # Viewer
        st.nt_fields = []
        st.nt_field = ""
        st.nt_cmap = "coolwarm"
        st.nt_cmaps = list(render.COLORMAPS)
        st.nt_show_edges = False
        st.nt_timestep = 0
        st.nt_nsteps = 1
        st.nt_has_timesteps = False

        # Analyze
        st.nt_method = "q_criterion"
        st.nt_method_choices = _ANALYSIS_CHOICES
        st.nt_analysis_done = False  # 파이프라인 칩 상태
        st.nt_fft_dt = 0.0  # 0 = 타임스텝에서 자동 추정
        st.nt_fft_summary = ""
        st.nt_fft_field = ""  # "" → _base_field() 사용
        st.nt_fft_probe = False  # False=공간 평균, True=프로브점
        st.nt_fft_point = 0

        # Charts (FFT/PSD, POD 에너지) — matplotlib PNG 모달
        st.nt_chart_dialog = False
        st.nt_chart_img = ""
        st.nt_chart_title = ""

        # Reduce
        st.nt_n_modes = 6
        st.nt_pod_done = False
        st.nt_pod_summary = ""
        st.nt_pod_energy = []
        st.nt_pod_mode = 0
        st.nt_pod_max_mode = 0

        # Model — 입출력 설정: 출력(예측 대상 필드)은 명시적으로 고른다 —
        # 3D 뷰어 컬러링용 nt_field 와 별개 (이전엔 몰래 재사용해 혼란을 줬다).
        # 입력은 항상 "시간(t)" — 단일 케이스 시계열이라 유일한 파라미터.
        # ROM 은 단일 출력(nt_train_field), Physics AI 는 다중 출력
        # (nt_train_fields — 한 신경망이 여러 필드를 동시 학습).
        st.nt_train_field = ""
        st.nt_train_fields = []
        st.nt_train_field_choices = []
        # 입력으로 쓸 다른 field (Physics AI 전용) — 주면 (좌표+입력장+시간)→출력
        # 의 field-to-field 연산자가 된다. 비면 좌표+시간만 입력.
        st.nt_train_input_fields = []

        # Model — 방식 우선(method-first) 2단 선택.
        # 문헌 4계열 중 웹 노출분: "rom"(Ⓐ 축소+보간) | "physics"(Ⓑ 직접 회귀)
        # | "operator"(Ⓒ 신경 연산자 — ⑥연산자 랩 으로 연결). 분류 근거는
        # .omc/plans/model-taxonomy-plan.md 참조.
        st.nt_model_method = "rom"
        st.nt_method_hint = ""  # 데이터 로드 시 service.recommend_method 결과
        # 능력 기반 전략 판정 (v5.0) — {key: {ok, reason, name}}. 카드가 이걸로
        # 가능/불가 + 이유를 데이터 로드 시점에 보여준다 (학습 버튼 눌러야 아는
        # 대신).
        st.nt_strategy_status = {}
        st.nt_reducer = "pod"
        st.nt_reducer_choices = [
            {"title": "POD (snapshot)", "value": "pod"},
            {"title": "Randomized POD", "value": "randomized_pod"},
        ]
        st.nt_surrogate = "rbf"
        st.nt_surrogate_choices = [
            {"title": "RBF", "value": "rbf"},
            {"title": "Kriging (GP)", "value": "kriging"},
            {"title": "GPR (EZyRB · 불확실성)", "value": "ezyrb_gpr"},
            {"title": "POD-NN (EZyRB 신경망)", "value": "ezyrb_ann"},
        ]
        st.nt_model_ready = False
        st.nt_model_summary = ""
        # surrogate="physicsnemo" 로 학습된 엔진인지 — ④Export 의 "PhysicsNeMo
        # Module" 버튼(실제 physicsnemo 패키지로 감싸기)은 이때만 의미가 있다.
        st.nt_physics_ready = False
        # Model — Physics AI (NVIDIA PhysicsNeMo) 학습 파라미터. POD reducer 없이
        # 좌표+시간 → 필드를 직접 매핑하는 PyTorch MLP (surrogate 선택이
        # "physicsnemo" 일 때만 쓰임 — Reducer 는 이때 숨겨진다).
        st.nt_physics_epochs = 150
        st.nt_physics_hidden = 32
        st.nt_physics_max_samples = 20_000

        # Model — 계열 Ⓓ 동역학 예보 (PyDMD). 학습 구간 밖 외삽이 가능한 유일
        # 계열. 적합도(재구성 오차)를 반드시 노출한다 — DMD 는 데이터가 안 맞으면
        # 조용히 크게 빗나간다. 근거: model-taxonomy-plan.md §20.
        st.nt_dmd_method = "dmd"
        st.nt_dmd_choices = [
            {"title": "DMD (표준)", "value": "dmd"},
            {"title": "Sparsity-promoting DMD", "value": "spdmd"},
        ]
        st.nt_dmd_ready = False
        st.nt_dmd_fit_error = 0.0
        st.nt_dmd_summary = ""
        # 학습 데이터의 t 상한. DMD 는 슬라이더 상한(nt_twin_max)을 이보다 크게
        # 잡아 외삽을 허용하므로, 이 값을 넘으면 UI 가 경고한다.
        st.nt_twin_train_max = 1.0

        # Twin — 문제 유형 A(시계열)는 스칼라 t 슬라이더 하나,
        # 문제 유형 B(케이스 세트)는 파라미터별 슬라이더 k 개(배열 상태).
        st.nt_twin_ready = False
        st.nt_twin_min = 0.0
        st.nt_twin_max = 1.0
        st.nt_twin_param = 0.0
        st.nt_twin_step = 0.01
        st.nt_twin_params = []
        st.nt_twin_mins = []
        st.nt_twin_maxs = []
        st.nt_twin_steps = []
        st.nt_twin_summary = ""
        # 출력 격자 자유화(M3): 비어 있으면 학습 격자, 값이 있으면 그 파일의
        # 메쉬 좌표에서 예측 중이라는 뜻 (Physics AI 전용 — 신경장이라 임의
        # 좌표 평가가 가능하다). 근거: model-taxonomy-plan.md §14, M3.
        st.nt_predict_mesh_name = ""

        # Compare (reducer×surrogate 벤치마크)
        st.nt_compare_dialog = False
        st.nt_compare_rows = []
        st.nt_compare_summary = ""
        st.nt_compare_headers = [
            {"title": "조합", "key": "combo"},
            {"title": "모드", "key": "n_modes", "align": "end"},
            {"title": "RMSE", "key": "rmse", "align": "end"},
            {"title": "R²", "key": "r2", "align": "end"},
            {"title": "nRMSE", "key": "rel_l2", "align": "end"},
            {"title": "지연(ms)", "key": "latency_ms", "align": "end"},
            {"title": "상태", "key": "status"},
        ]

        # AI Bench (벤치마크 데이터셋 → 연산자 학습)
        st.nt_bench_kind = "burgers"
        st.nt_bench_choices = [
            {"title": spec["title"], "value": key} for key, spec in bench.BENCHMARKS.items()
        ]
        st.nt_bench_nsamples = 64
        st.nt_bench_nx = 64
        st.nt_bench_path = ""
        st.nt_bench_ready = False
        st.nt_bench_summary = ""
        st.nt_bench_epochs = 60
        st.nt_bench_modes = 12
        st.nt_bench_width = 32
        # FNO 구현 백엔드 — 같은 계약이라 동일 벤치에서 직접 비교된다.
        # neuralop = FNO 논문 저자들이 유지하는 레퍼런스 구현(생태계 표준).
        st.nt_bench_backend = "neuralop"
        st.nt_bench_backend_choices = [
            {"title": "neuraloperator (레퍼런스)", "value": "neuralop"},
            {"title": "자체 구현 (builtin)", "value": "builtin"},
        ]
        st.nt_bench_trained = False
        st.nt_bench_train_summary = ""
        st.nt_bench_sample = 0
        st.nt_bench_max_sample = 0
        # 라이브 학습 진행 (StateQueue 로 워커에서 스트리밍)
        st.nt_bench_training = False
        st.nt_bench_epoch = 0
        st.nt_bench_epochs_total = 0
        st.nt_bench_loss = 0.0
        st.nt_bench_loss_series = []

        # Export
        import os.path as _osp

        st.nt_export_dir = _osp.join(_osp.expanduser("~"), "naviertwin-web")
        st.nt_export_last = ""

    def _register_callbacks(self) -> None:
        ctrl = self.ctrl
        # 무거운 연산은 비동기 래퍼로 — executor 에서 동기 워커 실행, 진행바 표시,
        # 렌더는 복귀 후 메인 스레드. 동기 워커(self.load_demo 등)는 테스트가 직접 호출.
        A = self._async
        ctrl.nt_load_demo = A(self.load_demo, "데모 데이터 로드 중…", render_after=True)
        ctrl.nt_load_path = A(self.load_path, "CFD 데이터 로드 중…", render_after=True)
        ctrl.nt_load_project = A(self.load_project, ".ntwin 프로젝트 로드 중…", render_after=True)
        # 파일 브라우저 (모달 열기/이동/선택은 즉시 반응 — 동기)
        ctrl.nt_coarsen = A(self.coarsen_current, "해상도 낮추는 중…", render_after=True)
        ctrl.nt_fb_open = self.fb_open
        ctrl.nt_fb_open_case_set = self.fb_open_case_set
        ctrl.nt_fb_open_predict_mesh = self.fb_open_predict_mesh
        ctrl.nt_fb_navigate = self.fb_navigate
        ctrl.nt_fb_pick = self.fb_pick
        ctrl.nt_fb_load_cwd = self.fb_load_cwd
        ctrl.nt_fb_load_case_set = self.fb_load_case_set
        ctrl.nt_restore_training_mesh = self.restore_training_mesh
        ctrl.nt_run_analysis = A(self.run_analysis, "와류 식별 계산 중…", render_after=True)
        ctrl.nt_run_fft = A(self.run_fft, "FFT/PSD 계산 중…")
        ctrl.nt_run_pod = A(self.run_pod, "POD 계산 중…")
        ctrl.nt_view_pod_mode = A(self.view_pod_mode, "POD 모드 렌더 중…", render_after=True)
        ctrl.nt_model_train = A(self.build_twin, "모델 학습 중…")
        ctrl.nt_build_twin = ctrl.nt_model_train  # 하위 호환 alias (비동기)
        # 비교/학습은 라이브 진행 전용 async(모니터 큐 push)로 — 진행바/스파크라인 스트리밍.
        ctrl.nt_run_compare = self._run_compare_async
        ctrl.nt_predict = A(self.predict, "예측 계산 중…", render_after=True)
        ctrl.nt_bench_generate = A(self.bench_generate, "벤치마크 데이터셋 생성 중…")
        ctrl.nt_bench_load = A(self.bench_load_h5, "PDEBench HDF5 로드 중…")
        ctrl.nt_bench_train = self._bench_train_async
        ctrl.nt_bench_eval = A(self.bench_evaluate, "샘플 예측 평가 중…")
        # 즉시 끝나거나 GL 을 메인 스레드에서 만져야 하는 콜백은 동기 유지.
        ctrl.nt_show_energy = self.show_energy_chart
        ctrl.nt_reset_view = self.reset_view
        ctrl.nt_export_screenshot = self.export_screenshot
        ctrl.nt_export_csv = self.export_csv
        ctrl.nt_export_vtk = self.export_vtk
        ctrl.nt_export_project = self.export_project
        ctrl.nt_export_engine = self.export_engine
        ctrl.nt_export_physicsnemo = self.export_physicsnemo
        ctrl.nt_export_report = self.export_report
        self.state.change("nt_field", "nt_cmap", "nt_show_edges", "nt_timestep")(
            self._on_view_state_change
        )
        # 케이스 슬라이더 — 시간축이 없는 케이스 세트의 "타임스텝" 역할.
        self.state.change("nt_case_index")(self.select_case)
        # 해상도를 고르는 동안 결과 크기를 계속 보여준다.
        self.state.change("nt_coarsen_resolution")(self._update_coarsen_preview)

    def _get_executor(self) -> Any:
        if self._executor is None:
            from concurrent.futures import ThreadPoolExecutor

            # 단일 사용자 — max_workers=1 로 무거운 연산을 직렬화(mesh 경쟁 방지).
            self._executor = ThreadPoolExecutor(max_workers=1)
        return self._executor

    async def _run_async(
        self,
        worker: Any,
        busy_message: str,
        *,
        render_after: bool = False,
    ) -> None:
        """동기 워커를 executor 에서 실행하며 진행바를 표시하고 렌더를 메인 스레드로 미룬다.

        중요(스레드 안전): 워커 스레드에서 ``with state:`` exit 시 일어나는
        ``state.flush()`` 는 웹소켓 push 와 ``@state.change`` 리스너(trame-vtk
        뷰 갱신 = VTK 접근 포함)를 **워커 스레드에서** 실행해 행(hang)을
        유발한다. 그래서 워커 실행 동안 trame 의 자체 ``flushing`` 가드를 켜
        워커 측 flush 를 무력화(상태 변경은 pending dict 에 축적만)하고,
        완료 후 이벤트 루프 스레드에서 한 번에 flush 한다.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        with self.state:
            self.state.nt_busy = True
            self.state.nt_status = busy_message
            self.state.nt_error = ""
        self.state.flush()  # nt_busy=True 를 브라우저에 즉시 푸시 (루프 스레드)

        self._defer_render = render_after
        status = getattr(self.state, "_status", None)  # trame-server StateStatus
        if status is not None:
            status.flushing = True  # 워커 스레드 flush 차단 (skip_flushing)
        try:
            await loop.run_in_executor(self._get_executor(), worker)
        finally:
            if status is not None:
                status.flushing = False
            self._defer_render = False
            with self.state:
                self.state.nt_busy = False
            self._flush_render()  # 보류된 렌더를 메인 스레드에서 수행
            self.state.flush()  # 워커가 축적한 상태를 루프 스레드에서 일괄 push

    def _async(self, worker: Any, message: str, *, render_after: bool = False) -> Any:
        """동기 워커를 감싸는 비동기 controller 콜백을 만든다."""

        async def _wrapped() -> None:
            await self._run_async(worker, message, render_after=render_after)

        return _wrapped

    # ------------------------------------------------------------------
    # Workflow callbacks
    # ------------------------------------------------------------------

    def load_demo(self) -> None:
        """선택한 데모를 로드한다 — 시계열이면 단일 데이터셋, 아니면 케이스 세트.

        데모 카탈로그(:data:`service.DEMO_CATALOG`)는 모델 계열별로 "잘 맞는
        데이터"를 하나씩 갖춘다 — 데이터가 없으면 기능을 시험할 수 없기 때문이다
        (예: DMD 는 ``waves`` 없이는 늘 크게 빗나간다).
        """
        kind = str(self.state.nt_demo_kind or "filament")
        title = next(
            (d["title"] for d in service.DEMO_CATALOG if d["value"] == kind), kind
        )
        # 해석 데모(karman*)도 기본 파라미터면 저장소 번들에서 즉시 로드된다 —
        # 앱은 항상 기본값으로만 부르므로 실제 계산 경로를 타지 않는다. 진행률
        # 배관을 여기서 켜지 않는다(모니터 생성은 loop 스레드 전용인데 이 함수는
        # 워커 스레드라 켜면 터진다).
        try:
            if kind in service.DEMO_CASE_SET_KINDS:
                result = service.make_demo_case_set(
                    kind, resolution=int(self.state.nt_case_resolution or 32)
                )
                self._set_case_set(result, f"데모 — {title}")
                return
            dataset = service.make_demo_dataset(kind=kind)
            self._set_dataset(dataset, status=f"데모 로드 완료 — {title}")
        except Exception as exc:  # noqa: BLE001
            self._fail("데모 로드 실패", exc)

    def load_path(self) -> None:
        """입력 경로에서 데이터를 로드한다 (.ntwin 이면 프로젝트로 디스패치)."""
        path = (self.state.nt_path or "").strip()
        if not path:
            self._fail("경로 필요", ValueError("CFD 파일/디렉토리 경로를 입력하세요."))
            return
        if path.lower().endswith(".ntwin"):
            self.load_project()
            return
        try:
            dataset = service.load_dataset(path)
            self._set_dataset(dataset, status=f"데이터 로드 완료: {path}")
        except Exception as exc:  # noqa: BLE001
            self._fail("데이터 로드 실패", exc)

    def load_project(self) -> None:
        """``.ntwin`` 프로젝트(+엔진 사이드카)를 로드해 워크플로우 상태를 복원한다."""
        path = (self.state.nt_path or "").strip()
        if not path:
            self._fail("경로 필요", ValueError(".ntwin 프로젝트 경로를 입력하세요."))
            return
        try:
            dataset, engine = service.load_project(path)
            self._set_dataset(dataset, status=f"프로젝트 로드 완료: {path}")
            if engine is not None:
                self._restore_engine(engine)
        except Exception as exc:  # noqa: BLE001
            self._fail("프로젝트 로드 실패", exc)

    def coarsen_current(self) -> None:
        """현재 데이터셋을 성긴 격자로 재샘플한다 (대용량 대응).

        학습·POD 는 (n_features × n_steps) 전체 행렬을 메모리에 올리므로,
        거대한 메쉬는 여기서 해상도를 낮춰야 실질적으로 다룰 수 있다. 되돌릴 수
        없으므로(원본 교체) 축소 결과를 상태에 남긴다.
        """
        if self.dataset is None:
            self._fail("데이터 없음", RuntimeError("먼저 데이터를 로드하세요."))
            return
        if self.state.nt_case_mode:
            self._fail(
                "해상도 낮추기",
                RuntimeError(
                    "케이스 세트는 로드 시 이미 공통 격자로 재샘플됩니다 — "
                    "해상도는 케이스 세트 로드 옵션으로 조절하세요."
                ),
            )
            return
        try:
            result = service.coarsen_dataset(
                self.dataset, resolution=int(self.state.nt_coarsen_resolution or 48)
            )
            summary = str(result["summary"])
            self._set_dataset(result["dataset"], status=f"해상도 낮춤: {summary}")
            with self.state:
                self.state.nt_coarsen_summary = summary
        except Exception as exc:  # noqa: BLE001
            self._fail("해상도 낮추기 실패", exc)

    def load_case_set(self) -> None:
        """폴더를 케이스 세트(정상 파라미터 스윕)로 로드한다 — 문제 유형 B."""
        path = (self.state.nt_path or "").strip()
        if not path:
            self._fail("경로 필요", ValueError("케이스 폴더를 선택하세요."))
            return
        try:
            # 메쉬가 서로 다르면 이 해상도의 공통 격자로 재샘플된다. 동일 메쉬면
            # 재샘플 자체가 생략되므로 이 값은 쓰이지 않는다.
            result = service.load_case_set(
                path, resolution=int(self.state.nt_case_resolution or 32)
            )
            self._set_case_set(result, path)
        except Exception as exc:  # noqa: BLE001
            self._fail("케이스 세트 로드 실패", exc)

    def _set_case_set(self, result: dict[str, Any], path: str) -> None:
        """케이스 세트를 적재한다 — 뷰어는 첫 케이스, 학습은 전체 케이스."""
        import numpy as np

        datasets = list(result["datasets"])
        names = list(result["param_names"])
        params = np.asarray(result["params"], dtype=float)
        # _set_dataset 이 케이스 상태를 리셋하므로 반드시 먼저 호출한다.
        self._set_dataset(datasets[0], status=f"케이스 세트 로드 완료: {path}")
        self.case_datasets = datasets
        self.case_params = params
        self.case_param_names = names

        mins = [float(v) for v in params.min(axis=0)]
        maxs = [float(v) for v in params.max(axis=0)]
        resampled = bool(result.get("resampled"))
        grid_summary = str(result.get("grid_summary") or "")
        with self.state:
            self.state.nt_case_mode = True
            self.state.nt_case_count = len(datasets)
            self.state.nt_case_names = list(result["case_names"])
            self.state.nt_params_source = str(result["params_source"])
            self.state.nt_param_names = names
            self.state.nt_case_index = 0
            # 케이스별 운전조건 문구 — 슬라이더로 넘길 때 무엇을 보는지 알려준다.
            self.state.nt_case_labels = [
                ", ".join(f"{n}={float(v):.4g}" for n, v in zip(names, row))
                for row in params
            ]
            self.state.nt_case_resampled = resampled
            self.state.nt_case_grid_summary = grid_summary
            shape_note = (
                " 케이스마다 메쉬가 달라 공통 격자로 재샘플했습니다(형상 가변) — "
                "sdf 필드로 형상 경계를 볼 수 있습니다."
                if resampled
                else ""
            )
            # 비정상 케이스 세트 (v5.0): 케이스마다 시계열이면 (μ, t) 로 학습된다.
            steps_per_case = max(
                (max(1, int(getattr(d, "n_time_steps", 1))) for d in datasets), default=1
            )
            time_note = (
                f" 케이스당 타임스텝 {steps_per_case}개 — 시간(t)도 입력 파라미터로 "
                "함께 학습됩니다 (비정상 스윕)."
                if steps_per_case > 1
                else " 정상 파라미터 스윕입니다."
            )
            self.state.nt_method_hint = (
                f"케이스 세트: {len(datasets)}개 케이스 × 입력 파라미터 {len(names)}개 "
                f"({', '.join(names)}) —{time_note} "
                f"스냅샷이 적으면 ROM 이 안정적입니다.{shape_note}"
            )
            try:
                self.state.nt_strategy_status = service.strategy_status(
                    datasets[0], datasets
                )
            except Exception:  # noqa: BLE001 — 판정은 부가 정보
                self.state.nt_strategy_status = {}
        self._set_twin_param_ranges(names, mins, maxs)
        # _set_dataset 은 케이스 모드를 켜기 전에 돌았으므로 단일 데이터셋 기준
        # 미리보기를 남긴다 — 케이스 세트에는 해당 없으니 지운다.
        self._update_coarsen_preview()

    def select_case(self, **_kwargs: Any) -> None:
        """케이스 세트에서 볼 케이스를 바꾼다 — 뷰어만 교체하고 학습 상태는 보존.

        케이스 세트는 시간축이 없어 타임스텝 슬라이더가 못 쓰이므로, 케이스별
        결과(원본 해)를 이 경로로 본다. ③Twin 예측 결과와 눈으로 비교할 수 있다.
        """
        if not self.case_datasets:
            return
        index = int(self.state.nt_case_index or 0)
        index = max(0, min(index, len(self.case_datasets) - 1))
        labels = list(self.state.nt_case_labels or [])
        names = list(self.state.nt_case_names or [])
        detail = labels[index] if index < len(labels) else ""
        name = names[index] if index < len(names) else f"#{index}"
        try:
            self._swap_view_dataset(
                self.case_datasets[index],
                status=f"케이스 {index + 1}/{len(self.case_datasets)} — {name} ({detail})",
            )
        except Exception as exc:  # noqa: BLE001
            self._fail("케이스 표시 실패", exc)

    def _set_twin_param_ranges(
        self, names: list[str], mins: list[float], maxs: list[float]
    ) -> None:
        """③Twin 의 파라미터별 슬라이더 상태(배열)를 설정한다 — 문제 유형 B."""
        with self.state:
            self.state.nt_param_names = list(names)
            self.state.nt_twin_mins = list(mins)
            self.state.nt_twin_maxs = list(maxs)
            self.state.nt_twin_params = [
                0.5 * (lo + hi) for lo, hi in zip(mins, maxs)
            ]
            self.state.nt_twin_steps = [
                max((hi - lo) / 100.0, 1e-6) for lo, hi in zip(mins, maxs)
            ]

    # ------------------------------------------------------------------
    # 파일 브라우저 (서버측 탐색기 모달)
    # ------------------------------------------------------------------

    def _fb_refresh(self, path: str | None = None) -> None:
        """디렉토리 목록을 갱신해 모달 상태에 반영한다."""
        listing = service.list_directory(path)
        st = self.state
        st.nt_fb_cwd = listing["cwd"]
        st.nt_fb_parent = listing["parent"]
        st.nt_fb_entries = listing["entries"]
        st.nt_fb_home = listing["home"]

    def fb_open(self) -> None:
        """파일 브라우저를 단일 파일/폴더 모드로 연다 (마지막 위치 또는 홈)."""
        self.state.nt_fb_mode = "single"
        self._fb_refresh((self.state.nt_fb_cwd or "").strip() or None)
        self.state.nt_fb_dialog = True

    def fb_open_case_set(self) -> None:
        """파일 브라우저를 케이스 세트 모드로 연다 (폴더 = 케이스 N개 + CSV)."""
        self.state.nt_fb_mode = "caseset"
        self._fb_refresh((self.state.nt_fb_cwd or "").strip() or None)
        self.state.nt_fb_dialog = True

    def fb_navigate(self, path: str) -> None:
        """모달 내에서 디렉토리를 이동한다."""
        self._fb_refresh(path)

    def _fb_dispatch_load(
        self, path: str, worker: Any = None, message: str = ""
    ) -> None:
        """모달을 닫고 경로 로드를 비동기로 트리거한다 (없으면 동기 폴백)."""
        import asyncio

        self.state.nt_path = path
        self.state.nt_fb_dialog = False
        if worker is None:
            worker = self.load_path
            is_proj = path.lower().endswith(".ntwin")
            message = ".ntwin 프로젝트 로드 중…" if is_proj else "CFD 데이터 로드 중…"
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            loop.create_task(self._run_async(worker, message, render_after=True))
        else:  # 테스트 등 루프 밖 컨텍스트
            worker()

    def fb_open_predict_mesh(self) -> None:
        """파일 브라우저를 "예측 격자 선택" 모드로 연다 (M3)."""
        self.state.nt_fb_mode = "predict_mesh"
        self._fb_refresh((self.state.nt_fb_cwd or "").strip() or None)
        self.state.nt_fb_dialog = True

    def fb_pick(self, path: str) -> None:
        """브라우저에서 파일을 선택 → 모드에 따라 로드/예측한다."""
        if self.state.nt_fb_mode == "predict_mesh":
            self._fb_dispatch_load(path, self.predict_on_mesh, "다른 격자에 예측 중…")
            return
        self._fb_dispatch_load(path)

    def fb_load_cwd(self) -> None:
        """현재 폴더 자체를 로드한다 (OpenFOAM case 디렉토리 등)."""
        cwd = (self.state.nt_fb_cwd or "").strip()
        if cwd:
            self._fb_dispatch_load(cwd)

    def fb_load_case_set(self) -> None:
        """현재 폴더를 케이스 세트로 로드한다 (파일 N개 = 운전조건 N개)."""
        cwd = (self.state.nt_fb_cwd or "").strip()
        if cwd:
            self._fb_dispatch_load(cwd, self.load_case_set, "케이스 세트 로드 중…")

    def _restore_engine(self, engine: Any) -> None:
        """로드된 TwinEngine 으로 Model/Twin 상태를 복원한다.

        예측 슬라이더 범위는 엔진 metadata 의 학습 파라미터 범위를 최우선으로
        사용한다 — dataset 시간축 fallback 은 단일 스냅샷 프로젝트에서 학습
        범위 밖(외삽) 슬라이더를 만들 수 있기 때문.
        """
        self.engine = engine
        meta = getattr(engine, "training_metadata", {}) or {}
        field = meta.get("field_name", "")
        reducer = meta.get("reducer", getattr(engine, "reducer_type", "pod"))
        surrogate = meta.get("surrogate", getattr(engine, "surrogate_type", "rbf"))
        if meta.get("problem_type") == "steady_sweep":
            self._restore_sweep_engine(engine, meta, reducer, surrogate, field)
            return
        meta_min = meta.get("param_min")
        meta_max = meta.get("param_max")
        times = [float(t) for t in (getattr(self.dataset, "time_steps", []) or [])]
        if (
            isinstance(meta_min, (int, float))
            and isinstance(meta_max, (int, float))
            and float(meta_max) > float(meta_min)
        ):
            pmin, pmax = float(meta_min), float(meta_max)
        elif len(times) >= 2:
            pmin, pmax = min(times), max(times)
        elif len(times) == 1:
            pmin, pmax = times[0], times[0] + 1.0  # zero-width 슬라이더 방지
        else:
            pmin, pmax = 0.0, 1.0
        summary = f"복원된 엔진: {reducer}+{surrogate}, field='{field}'"
        with self.state:
            if str(reducer) == "direct_physics_ai":
                self.state.nt_model_method = "physics"
            restored_fields = meta.get("field_names") or ([field] if field else [])
            if restored_fields:
                self.state.nt_train_fields = list(restored_fields)
                self.state.nt_train_field = str(restored_fields[0])
            self.state.nt_model_ready = True
            self.state.nt_twin_ready = True
            self.state.nt_twin_min = pmin
            self.state.nt_twin_max = pmax
            self.state.nt_twin_param = 0.5 * (pmin + pmax)
            self.state.nt_twin_step = max((pmax - pmin) / 100.0, 1e-6)
            self.state.nt_model_summary = summary
            self.state.nt_twin_summary = summary
        self._set_status(f"프로젝트 로드 완료 — {summary}")

    def _restore_sweep_engine(
        self, engine: Any, meta: dict[str, Any], reducer: Any, surrogate: Any, field: Any
    ) -> None:
        """케이스 세트(파라미터 스윕) 엔진을 복원한다 — 파라미터별 슬라이더 k 개.

        프로젝트 파일은 케이스 1개(첫 케이스 스냅샷)만 담으므로 재학습은 못
        하지만, 저장된 파라미터 범위로 예측은 그대로 가능하다.
        """
        names = [str(n) for n in (meta.get("param_names") or [])]
        mins = [float(v) for v in (meta.get("param_mins") or [])]
        maxs = [float(v) for v in (meta.get("param_maxs") or [])]
        if not (names and len(names) == len(mins) == len(maxs)):
            self._fail(
                "엔진 복원",
                RuntimeError("파라미터 스윕 엔진의 파라미터 메타데이터가 불완전합니다."),
            )
            return
        summary = (
            f"복원된 엔진: {reducer}+{surrogate}, field='{field}' · "
            f"입력 파라미터 {len(names)}개 ({', '.join(names)})"
        )
        with self.state:
            if str(reducer) == "direct_physics_ai":
                self.state.nt_model_method = "physics"
            restored_fields = meta.get("field_names") or ([field] if field else [])
            if restored_fields:
                self.state.nt_train_fields = list(restored_fields)
                self.state.nt_train_field = str(restored_fields[0])
            # 예측 전용 — 케이스 1개만 복원되므로 재학습은 불가(case_datasets 없음).
            self.state.nt_case_mode = True
            self.state.nt_model_ready = True
            self.state.nt_twin_ready = True
            self.state.nt_model_summary = summary
            self.state.nt_twin_summary = summary
        self._set_twin_param_ranges(names, mins, maxs)
        self._set_status(f"프로젝트 로드 완료 — {summary}")

    def run_analysis(self) -> None:
        """현재 데이터셋에 와류 식별(Q/λ₂)을 실행한다."""
        if self.dataset is None:
            self._fail("데이터 없음", RuntimeError("먼저 데이터를 로드하세요."))
            return
        method = self.state.nt_method or "q_criterion"
        try:
            field = service.run_vortex_analysis(
                self.dataset, method, timestep=int(self.state.nt_timestep or 0)
            )
            self._refresh_fields(prefer=field)
            self._render(reset_camera=False)
            with self.state:
                self.state.nt_analysis_done = True
            self._set_status(f"분석 완료: {service.RESULT_FIELD.get(method, method)} → {field}")
        except Exception as exc:  # noqa: BLE001
            self._fail("분석 실패", exc)

    def run_fft(self) -> None:
        """선택 field 시계열의 FFT/PSD 를 계산하고 차트 모달로 표시한다."""
        if self.dataset is None:
            self._fail("데이터 없음", RuntimeError("먼저 데이터를 로드하세요."))
            return
        field = (self.state.nt_fft_field or "").strip() or self._base_field()
        try:
            dt_value = float(self.state.nt_fft_dt or 0.0)
            point_index = int(self.state.nt_fft_point or 0) if bool(self.state.nt_fft_probe) else None
            result = service.compute_fft_psd(
                self.dataset,
                field,
                dt=dt_value if dt_value > 0 else None,
                point_index=point_index,
            )
            peaks = result["dominant"]
            peak_text = ", ".join(
                f"{p['frequency']:.3g}Hz" for p in peaks[:3]
            ) or "뚜렷한 피크 없음"
            with self.state:
                self.state.nt_fft_summary = (
                    f"field='{field}', dt={result['dt']:.4g}s · 지배 주파수: {peak_text}"
                )
            self._show_chart(
                self._plot_fft(result),
                title=f"FFT / PSD — {field} ({result['probe']})",
            )
            self._set_status(f"FFT/PSD 완료: {peak_text}")
        except Exception as exc:  # noqa: BLE001
            self._fail("FFT/PSD 실패", exc)

    def run_pod(self) -> None:
        """선택 field 에 POD 차원 축소를 수행하고 에너지 곡선을 표시한다."""
        if self.dataset is None:
            self._fail("데이터 없음", RuntimeError("먼저 데이터를 로드하세요."))
            return
        field = self._base_field()
        try:
            result = service.run_pod(self.dataset, field, int(self.state.nt_n_modes or 6))
            self.reducer = result["reducer"]
            self._pod_result = result
            energy = result["cumulative_energy"]
            with self.state:
                self.state.nt_pod_done = True
                self.state.nt_pod_max_mode = max(0, int(result["n_modes"]) - 1)
                self.state.nt_pod_energy = [round(float(e) * 100.0, 2) for e in energy]
                self.state.nt_pod_summary = (
                    f"field='{field}', 모드 {result['n_modes']}개 / "
                    f"스냅샷 {result['n_snapshots']}개 · "
                    f"누적 에너지 {energy[-1] * 100:.2f}%"
                )
            self._set_status(f"POD 완료: {result['n_modes']} 모드, 에너지 {energy[-1] * 100:.1f}%")
        except Exception as exc:  # noqa: BLE001
            self._fail("POD 실패", exc)

    def view_pod_mode(self) -> None:
        """학습된 POD 모드 형상을 3D 뷰어에 표시한다."""
        if self.dataset is None or self.reducer is None:
            self._fail("POD 없음", RuntimeError("먼저 ⑤부가 분석에서 POD 를 실행하세요."))
            return
        try:
            index = max(0, min(int(self.state.nt_pod_mode or 0), int(self.state.nt_pod_max_mode or 0)))
            field = service.attach_pod_mode(self.dataset, self.reducer, index)
            self._refresh_fields(prefer=field)
            self._render(reset_camera=False)
            self._set_status(f"POD 모드 #{index} 3D 표시: '{field}'")
        except Exception as exc:  # noqa: BLE001
            self._fail("POD 모드 표시 실패", exc)

    def show_energy_chart(self) -> None:
        """POD 특이값 + 누적 에너지 스펙트럼을 차트 모달로 표시한다."""
        if not self._pod_result:
            self._fail("POD 없음", RuntimeError("먼저 ⑤부가 분석에서 POD 를 실행하세요."))
            return
        try:
            self._show_chart(
                self._plot_energy(self._pod_result),
                title=f"POD 에너지 스펙트럼 — {self._pod_result.get('field', '')}",
            )
            self._set_status("POD 에너지 스펙트럼 표시")
        except Exception as exc:  # noqa: BLE001
            self._fail("에너지 차트 실패", exc)

    # ------------------------------------------------------------------
    # Charts (matplotlib → base64 PNG → trame 모달)
    # ------------------------------------------------------------------

    def _show_chart(self, image_uri: str, *, title: str) -> None:
        with self.state:
            self.state.nt_chart_img = image_uri
            self.state.nt_chart_title = title
            self.state.nt_chart_dialog = True

    @staticmethod
    def _figure_to_uri(fig: Any) -> str:
        """matplotlib Figure 를 base64 data URI(PNG)로 변환한다 (실패해도 figure 는 닫는다)."""
        import base64
        from io import BytesIO

        import matplotlib.pyplot as plt

        buffer = BytesIO()
        try:
            fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
        finally:
            plt.close(fig)  # savefig 실패 시에도 figure 누수 방지
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @_dark_chart
    def _plot_fft(self, result: dict[str, Any]) -> str:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, (ax_amp, ax_psd) = plt.subplots(2, 1, figsize=(7.2, 5.4))
        ax_amp.plot(result["freqs"], result["amplitudes"], color=theme.PRIMARY)
        ax_amp.set_title("FFT amplitude spectrum")
        ax_amp.set_xlabel("Frequency [Hz]")
        ax_amp.set_ylabel("Amplitude")
        ax_amp.grid(True, alpha=0.3)
        for peak in result["dominant"][:3]:
            ax_amp.axvline(peak["frequency"], color=theme.WARNING, ls="--", lw=1)
        ax_psd.semilogy(result["psd_freqs"], result["psd"], color=theme.SECONDARY)
        ax_psd.set_title("Power spectral density (Welch)")
        ax_psd.set_xlabel("Frequency [Hz]")
        ax_psd.set_ylabel("PSD")
        ax_psd.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        return self._figure_to_uri(fig)

    @_dark_chart
    def _plot_energy(self, result: dict[str, Any]) -> str:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        singular = result.get("singular_values", [])
        energy = result.get("cumulative_energy", [])
        modes = list(range(1, len(singular) + 1))
        fig, ax_sv = plt.subplots(figsize=(7.2, 4.2))
        ax_sv.bar(modes, singular, color=theme.PRIMARY, alpha=0.85, label="Singular value")
        ax_sv.set_xlabel("Mode")
        ax_sv.set_ylabel("Singular value", color=theme.PRIMARY)
        ax_sv.grid(True, axis="y", alpha=0.3)
        ax_energy = ax_sv.twinx()
        ax_energy.plot(
            modes,
            [e * 100.0 for e in energy],
            color=theme.WARNING,
            marker="o",
            label="Cumulative energy",
        )
        ax_energy.set_ylabel("Cumulative energy [%]", color=theme.WARNING)
        ax_energy.set_ylim(0, 105)
        ax_sv.set_title("POD spectrum")
        fig.tight_layout()
        return self._figure_to_uri(fig)

    @_dark_chart
    def _plot_bench_loss(self, result: dict[str, Any]) -> str:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        losses = result.get("losses", [])
        op = str(result.get("operator", "fno")).upper()
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.semilogy(range(1, len(losses) + 1), losses, color=theme.PRIMARY)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train MSE loss")
        ax.set_title(
            f"{op} training — test RMSE {result.get('test_rmse', float('nan')):.3g}, "
            f"inference {result.get('latency_ms', float('nan')):.2f} ms"
        )
        ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        return self._figure_to_uri(fig)

    @_dark_chart
    def _plot_bench_eval(self, ev: dict[str, Any]) -> str:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        title = (
            f"Sample #{ev['index']} — RMSE {ev['rmse']:.3g}, latency {ev['latency_ms']:.2f} ms"
        )

        if ev.get("is_2d"):
            fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
            panels = (
                ("true |u|", ev["true2d"], "viridis"),
                ("FNO pred |u|", ev["pred2d"], "viridis"),
                ("|error|", ev["err2d"], "magma"),
            )
            panel_idx = 0
            while panel_idx < len(panels):
                label, data, cmap = panels[panel_idx]
                ax = axes[panel_idx]
                im = ax.imshow(data, origin="lower", cmap=cmap)
                ax.set_title(label, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046)
                panel_idx += 1
            fig.suptitle(title, fontsize=11)
            fig.tight_layout()
            return self._figure_to_uri(fig)

        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.plot(ev["x"], ev["input"], color=theme.MUTED, ls=":", lw=1.2, label="input u0")
        ax.plot(ev["x"], ev["true"], color=theme.SECONDARY, lw=2.0, label="true u(T)")
        ax.plot(ev["x"], ev["pred"], color=theme.WARNING, ls="--", lw=1.8, label="FNO pred")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self._figure_to_uri(fig)

    def build_twin(self) -> None:
        """②Model — 선택한 방식으로 (입력 → 필드) 트윈을 학습한다.

        ``nt_model_method`` 로 디스패치한다: "physics" 는 POD reducer 없이
        좌표+입력→필드를 직접 학습(:meth:`_build_physics_twin`), "operator" 는
        ⑥연산자 랩 안내, 기본("rom")은 POD reducer + 계수 회귀(rbf/kriging).
        입력은 문제 유형에 따라 시간(t) 또는 운전조건 벡터(케이스 세트)다.
        """
        if self.dataset is None:
            self._fail("데이터 없음", RuntimeError("먼저 데이터를 로드하세요."))
            return
        if self.state.nt_predict_mesh_name and not self.state.nt_case_mode:
            # 뷰어가 예측 격자로 바뀐 상태라 self.dataset 이 학습 데이터가 아니다.
            self._fail(
                "예측 격자 모드",
                RuntimeError("'학습 격자로 복귀' 후 재학습하세요."),
            )
            return
        method = self.state.nt_model_method or "rom"
        surrogate = self.state.nt_surrogate or "rbf"
        # 하위호환 shim: 한때 surrogate 값으로 physicsnemo 를 표현했다.
        if method == "physics" or surrogate == "physicsnemo":
            self._build_physics_twin()
            return
        if method == "operator":
            self._fail(
                "신경 연산자",
                RuntimeError("신경 연산자(FNO) 학습은 ⑥연산자 랩 패널에서 실행하세요."),
            )
            return
        if method == "dynamics":
            if self.state.nt_case_mode:
                # 비정상 케이스 세트 → ParametricDMD (v5.2). 뷰어 데이터셋(첫
                # 케이스)만으로 단일 DMD 가 조용히 "성공"하는 함정을 막고 전체
                # 케이스로 학습한다.
                self._build_parametric_dmd_twin()
                return
            self._build_dmd_twin()
            return
        if self.state.nt_case_mode:
            self._build_sweep_twin()
            return
        field = self._training_field()
        reducer = self.state.nt_reducer or "pod"
        try:
            result = service.build_twin(
                self.dataset,
                field,
                int(self.state.nt_n_modes or 6),
                reducer=reducer,
                surrogate=surrogate,
            )
            self.engine = result["engine"]
            pmin, pmax = result["param_min"], result["param_max"]
            summary = (
                f"field='{field}', {reducer}+{surrogate}, 모드 {result['n_modes']}개 · "
                f"파라미터(t) ∈ [{pmin:.3g}, {pmax:.3g}]"
            )
            with self.state:
                self.state.nt_model_ready = True
                self.state.nt_model_summary = summary
                self.state.nt_physics_ready = False
                self.state.nt_dmd_ready = False
                self.state.nt_twin_ready = True
                self.state.nt_twin_min = pmin
                self.state.nt_twin_max = pmax
                self.state.nt_twin_train_max = pmax  # 내삽 전용 — 외삽 구간 없음
                self.state.nt_twin_param = 0.5 * (pmin + pmax)
                self.state.nt_twin_step = max((pmax - pmin) / 100.0, 1e-6)
                self.state.nt_twin_summary = summary
            self._set_status(f"모델 학습 완료: {reducer}+{surrogate}. ③Twin 에서 예측하세요.")
        except Exception as exc:  # noqa: BLE001
            self._fail("모델 학습 실패", exc)

    def _build_parametric_dmd_twin(self) -> None:
        """②Model — 비정상 케이스 세트의 (μ, t) 예보 트윈 (ParametricDMD, v5.2).

        학습에 없던 μ 는 모달 계수를 보간하고, t 는 학습 구간 밖까지 예보한다.
        적합도(재구성 오차)는 단일 DMD 와 같은 트래픽 라이트로 표시한다.
        """
        if not self.case_datasets:
            self._fail(
                "케이스 없음",
                RuntimeError("케이스 폴더를 다시 로드하세요."),
            )
            return
        field = self._training_field()
        try:
            result = service.build_parametric_dmd_twin(
                self.case_datasets,
                field,
                self.case_params,
                param_names=self.case_param_names,
            )
            self.engine = result["engine"]
            names = result["param_names"]
            err = result["reconstruction_error"]
            summary = (
                f"field='{field}', ParametricDMD(partitioned) · 케이스 "
                f"{result['n_cases']}개 · 입력 ({', '.join(names)}) · "
                f"학습 t ≤ {result['train_t_max']:.3g} · "
                f"예보 t ≤ {result['forecast_t_max']:.3g}"
            )
            with self.state:
                self.state.nt_model_ready = True
                self.state.nt_model_summary = summary
                self.state.nt_physics_ready = False
                self.state.nt_dmd_ready = True
                self.state.nt_dmd_fit_error = float(err)
                self.state.nt_dmd_summary = summary
                self.state.nt_twin_ready = True
                self.state.nt_twin_summary = summary
            self._set_twin_param_ranges(names, result["param_mins"], result["param_maxs"])
            self._set_status(
                f"ParametricDMD 학습 완료 (재구성 오차 {err * 100:.1f}%). "
                "③Twin 에서 학습에 없던 운전조건·미래 t 를 예보하세요."
            )
        except Exception as exc:  # noqa: BLE001
            self._fail("ParametricDMD 학습 실패", exc)

    def _build_dmd_twin(self) -> None:
        """②Model — PyDMD 동역학 트윈을 학습한다 (계열 Ⓓ, 시간 외삽 가능).

        케이스 세트(문제 유형 B)는 시간축이 없어 적용 불가 — 시계열 전용이다.
        """
        if self.state.nt_case_mode:
            self._fail(
                "동역학 예보",
                RuntimeError(
                    "동역학 예보는 시계열 데이터 전용입니다 — 케이스 세트"
                    "(파라미터 스윕)에는 시간축이 없습니다."
                ),
            )
            return
        field = self._training_field()
        method = self.state.nt_dmd_method or "dmd"
        try:
            result = service.build_dmd_twin(self.dataset, field, method=method)
            self.engine = result["engine"]
            pmin, pmax = result["param_min"], result["param_max"]
            fmax = result["forecast_max"]
            err = result["reconstruction_error"]
            summary = (
                f"field='{field}', DMD({method}) · 모드 {result['n_modes']}개 · "
                f"학습 t ∈ [{pmin:.3g}, {pmax:.3g}] · 예보 t ≤ {fmax:.3g}"
            )
            with self.state:
                self.state.nt_model_ready = True
                self.state.nt_model_summary = summary
                self.state.nt_physics_ready = False
                self.state.nt_dmd_ready = True
                self.state.nt_dmd_fit_error = float(err)
                self.state.nt_dmd_summary = summary
                self.state.nt_twin_ready = True
                self.state.nt_twin_min = pmin
                # 슬라이더 상한을 학습 구간 밖까지 — 여기가 이 계열의 존재 이유.
                self.state.nt_twin_max = fmax
                self.state.nt_twin_train_max = pmax  # 경고 임계 = 학습 t 상한
                self.state.nt_twin_param = pmax
                self.state.nt_twin_step = max((fmax - pmin) / 200.0, 1e-6)
                self.state.nt_twin_summary = summary
            self._set_status(
                f"DMD 학습 완료 (재구성 오차 {err * 100:.1f}%). "
                "③Twin 에서 학습 구간 밖까지 예보하세요."
            )
        except Exception as exc:  # noqa: BLE001
            self._fail("DMD 학습 실패", exc)

    def _build_sweep_twin(self) -> None:
        """②Model — 케이스 세트에서 (운전조건 → 필드) ROM 트윈을 학습한다.

        문제 유형 B — 입력이 시간이 아니라 파라미터 CSV 의 k 차원 운전조건이다.
        """
        if not self.case_datasets:
            self._fail(
                "케이스 없음",
                RuntimeError(
                    "복원된 프로젝트는 케이스 1개만 담아 재학습할 수 없습니다 — "
                    "케이스 폴더를 다시 로드하세요."
                ),
            )
            return
        field = self._training_field()
        reducer = self.state.nt_reducer or "pod"
        surrogate = self.state.nt_surrogate or "rbf"
        try:
            result = service.build_twin_from_cases(
                self.case_datasets,
                field,
                int(self.state.nt_n_modes or 6),
                self.case_params,
                param_names=self.case_param_names,
                reducer=reducer,
                surrogate=surrogate,
            )
            self.engine = result["engine"]
            names = result["param_names"]
            summary = (
                f"field='{field}', {reducer}+{surrogate}, 모드 {result['n_modes']}개 · "
                f"케이스 {result['n_cases']}개 · 입력 파라미터 {len(names)}개 "
                f"({', '.join(names)})"
            )
            with self.state:
                self.state.nt_model_ready = True
                self.state.nt_model_summary = summary
                self.state.nt_physics_ready = False
                self.state.nt_dmd_ready = False
                self.state.nt_twin_ready = True
                self.state.nt_twin_summary = summary
            self._set_twin_param_ranges(names, result["param_mins"], result["param_maxs"])
            self._set_status(
                f"모델 학습 완료: {reducer}+{surrogate} (파라미터 스윕). "
                "③Twin 에서 예측하세요."
            )
        except Exception as exc:  # noqa: BLE001
            self._fail("모델 학습 실패", exc)

    def _build_physics_twin(self) -> None:
        """②Model — NVIDIA PhysicsNeMo 스타일 직접 필드 예측 모델을 학습한다.

        POD reducer 없이 (좌표+입력) → 필드를 곧장 학습하는 PyTorch MLP.
        결과 엔진은 ``predict()``만 노출하는 TwinEngine 과 같은 계약이라
        ③Twin/④Export 는 수정 없이 그대로 재사용된다. 케이스 세트(문제 유형 B)
        면 입력이 (좌표 + 운전조건 벡터)로 확장된다.
        """
        if self.state.nt_case_mode:
            self._build_sweep_physics_twin()
            return
        fields = self._training_fields()
        inputs = self._training_input_fields()
        try:
            result = service.build_physics_ai_twin(
                self.dataset,
                fields,
                input_fields=inputs,
                hidden=int(self.state.nt_physics_hidden or 32),
                max_epochs=int(self.state.nt_physics_epochs or 150),
                max_train_points=int(self.state.nt_physics_max_samples or 20_000),
            )
            self.engine = result["engine"]
            pmin, pmax = result["param_min"], result["param_max"]
            metrics = result["validation_metrics"]
            rmse = metrics.get("rmse", float("nan"))
            label = ", ".join(fields)
            multi = f", 다중 출력 {len(fields)}개" if len(fields) > 1 else ""
            in_label = f"입력 {', '.join(inputs)}+시간(t)" if inputs else "입력 시간(t)"
            summary = (
                f"field(s)='{label}', PhysicsNeMo CFD Field (직접 예측{multi}) · "
                f"{in_label} · 범위 t ∈ [{pmin:.3g}, {pmax:.3g}] · RMSE {rmse:.3g}"
            )
            with self.state:
                self.state.nt_model_ready = True
                self.state.nt_model_summary = summary
                self.state.nt_physics_ready = True
                self.state.nt_dmd_ready = False
                self.state.nt_twin_ready = True
                self.state.nt_twin_train_max = pmax  # 내삽 전용
                self.state.nt_twin_min = pmin
                self.state.nt_twin_max = pmax
                self.state.nt_twin_param = 0.5 * (pmin + pmax)
                self.state.nt_twin_step = max((pmax - pmin) / 100.0, 1e-6)
                self.state.nt_twin_summary = summary
            self._set_status("PhysicsNeMo 모델 학습 완료. ③Twin 에서 예측하세요.")
        except Exception as exc:  # noqa: BLE001
            self._fail("PhysicsNeMo 학습 실패", exc)

    def _build_sweep_physics_twin(self) -> None:
        """②Model — 케이스 세트에서 (좌표+운전조건 → 필드) Physics AI 를 학습한다."""
        if not self.case_datasets:
            self._fail(
                "케이스 없음",
                RuntimeError(
                    "복원된 프로젝트는 케이스 1개만 담아 재학습할 수 없습니다 — "
                    "케이스 폴더를 다시 로드하세요."
                ),
            )
            return
        fields = self._training_fields()
        try:
            result = service.build_physics_ai_twin_from_cases(
                self.case_datasets,
                fields,
                self.case_params,
                param_names=self.case_param_names,
                hidden=int(self.state.nt_physics_hidden or 32),
                max_epochs=int(self.state.nt_physics_epochs or 150),
                max_train_points=int(self.state.nt_physics_max_samples or 20_000),
            )
            self.engine = result["engine"]
            names = result["param_names"]
            rmse = result["validation_metrics"].get("rmse", float("nan"))
            multi = f", 다중 출력 {len(fields)}개" if len(fields) > 1 else ""
            summary = (
                f"field(s)='{', '.join(fields)}', PhysicsNeMo CFD Field "
                f"(직접 예측{multi}) · 케이스 {result['n_cases']}개 · 입력 파라미터 "
                f"{len(names)}개 ({', '.join(names)}) · RMSE {rmse:.3g}"
            )
            with self.state:
                self.state.nt_model_ready = True
                self.state.nt_model_summary = summary
                self.state.nt_physics_ready = True
                self.state.nt_dmd_ready = False
                self.state.nt_twin_ready = True
                self.state.nt_twin_summary = summary
            self._set_twin_param_ranges(names, result["param_mins"], result["param_maxs"])
            self._set_status(
                "PhysicsNeMo 모델 학습 완료 (파라미터 스윕). ③Twin 에서 예측하세요."
            )
        except Exception as exc:  # noqa: BLE001
            self._fail("PhysicsNeMo 학습 실패", exc)

    def run_compare(self) -> None:
        """모든 reducer×surrogate 조합을 학습/평가해 순위표를 표시한다."""
        if self.dataset is None:
            self._fail("데이터 없음", RuntimeError("먼저 데이터를 로드하세요."))
            return
        if self.state.nt_case_mode:
            # 리더보드는 시계열 경로(compare_models)만 지원 — 케이스 세트 확장은
            # 후속(M2 follow-up). 애매한 내부 에러 대신 명확히 알린다.
            self._fail(
                "자동 비교",
                RuntimeError("케이스 세트(파라미터 스윕)의 자동 비교는 아직 지원하지 않습니다."),
            )
            return
        try:
            summary = self._apply_compare_result(self._compare_compute(None))
            self._set_status(f"모델 비교 완료 — {summary}")
        except Exception as exc:  # noqa: BLE001
            self._fail("모델 비교 실패", exc)

    def _compare_compute(self, progress_cb: Any) -> dict[str, Any]:
        """전체 reducer×surrogate 조합 벤치마크 (동기 워커 — 상태 미변경)."""
        field = self._training_field()
        return service.compare_models(
            self.dataset, field, int(self.state.nt_n_modes or 6), progress_cb=progress_cb
        )

    def _apply_compare_result(self, result: dict[str, Any]) -> str:
        """비교 결과를 표/상태에 반영한다 (loop 스레드에서만 호출)."""
        rows = [self._format_compare_row(row) for row in result["rows"]]
        best = result["best"]
        field = result.get("field", "")
        summary = (
            f"field='{field}' · 최우수(RMSE): {best['combo']}"
            if best
            else f"field='{field}' · 유효한 조합 없음"
        )
        with self.state:
            self.state.nt_compare_rows = rows
            self.state.nt_compare_summary = summary
            self.state.nt_compare_dialog = True
        return summary

    async def _run_compare_async(self) -> None:
        """모델 비교 (라이브 진행 — 조합별 진행바 스트리밍)."""
        import asyncio

        if self.dataset is None:
            self._fail("데이터 없음", RuntimeError("먼저 데이터를 로드하세요."))
            return
        loop = asyncio.get_running_loop()
        self._ensure_progress_monitor()

        def progress_cb(done: int, total: int, label: str) -> None:
            self._progress_queue.put_nowait(
                {
                    "nt_progress": 100.0 * done / max(1, total),
                    "nt_status": f"모델 비교 중… ({done}/{total}) {label}",
                }
            )

        with self.state:
            self.state.nt_busy = True
            self.state.nt_error = ""
            self.state.nt_progress = 0.0
            self.state.nt_status = "모델 비교 중…"
        self.state.flush()
        try:
            result = await loop.run_in_executor(
                self._get_executor(), lambda: self._compare_compute(progress_cb)
            )
        except Exception as exc:  # noqa: BLE001
            with self.state:
                self.state.nt_progress = -1.0
            self._fail("모델 비교 실패", exc)
            return
        summary = self._apply_compare_result(result)
        self._progress_queue.put_nowait({"nt_progress": -1.0})
        with self.state:
            self.state.nt_busy = False
        self._set_status(f"모델 비교 완료 — {summary}")

    @staticmethod
    def _format_compare_row(row: dict[str, Any]) -> dict[str, Any]:
        """compare_models 결과 행을 VDataTable 표시용으로 포맷한다 (inf/nan → '—')."""
        import math

        def fmt(value: Any, spec: str = "{:.4g}") -> str:
            if value is None or (isinstance(value, float) and (math.isinf(value) or math.isnan(value))):
                return "—"
            return spec.format(value)

        status = str(row.get("status", ""))
        n_modes = int(row.get("n_modes", 0))
        return {
            "combo": row.get("combo", ""),
            # Physics AI(직접 회귀)는 모드 개념이 없다 → 0 대신 '—'.
            "n_modes": n_modes if n_modes > 0 else "—",
            "rmse": fmt(row.get("rmse")),
            "r2": fmt(row.get("r2"), "{:.4f}"),
            "rel_l2": fmt(row.get("rel_l2")),
            "latency_ms": fmt(row.get("latency_ms"), "{:.3f}"),
            "status": "ok" if status == "ok" else status[:40],
        }

    # ------------------------------------------------------------------
    # 연산자 랩 callbacks (⑦, 구 AI Bench) — 벤치마크 데이터셋 → 연산자 학습
    # ------------------------------------------------------------------

    def _bench_set_dataset(self, dataset: dict[str, Any], *, status: str) -> None:
        """정규화된 연산자 데이터셋을 상태에 반영한다 (CFD 워크플로우와 독립)."""
        self._bench_dataset = dataset
        self._bench_result = None
        with self.state:
            self.state.nt_bench_ready = True
            self.state.nt_bench_summary = bench.dataset_summary(dataset)
            self.state.nt_bench_trained = False
            self.state.nt_bench_train_summary = ""
            self.state.nt_bench_sample = 0
            self.state.nt_bench_max_sample = max(0, int(dataset.get("n_samples", 1)) - 1)
            self.state.nt_bench_training = False
            self.state.nt_bench_loss_series = []
            self.state.nt_bench_epoch = 0
        self._set_status(status)

    def bench_generate(self) -> None:
        """내장 솔버로 (u0→uT) 벤치마크 데이터셋을 생성한다."""
        kind = self.state.nt_bench_kind or "burgers"
        try:
            dataset = bench.generate_operator_dataset(
                kind,
                n_samples=int(self.state.nt_bench_nsamples or 64),
                n_x=int(self.state.nt_bench_nx or 64),
            )
            self._bench_set_dataset(
                dataset, status=f"벤치마크 데이터셋 생성 완료: {bench.dataset_summary(dataset)}"
            )
        except Exception as exc:  # noqa: BLE001
            self._fail("데이터셋 생성 실패", exc)

    def bench_load_h5(self) -> None:
        """PDEBench 포맷 HDF5 파일을 연산자 데이터셋으로 로드한다."""
        path = (self.state.nt_bench_path or "").strip()
        if not path:
            self._fail("경로 필요", ValueError("PDEBench .h5/.hdf5 파일 경로를 입력하세요."))
            return
        try:
            dataset = bench.load_pdebench_hdf5(path)
            self._bench_set_dataset(
                dataset, status=f"PDEBench 로드 완료: {bench.dataset_summary(dataset)}"
            )
        except Exception as exc:  # noqa: BLE001
            self._fail("PDEBench 로드 실패", exc)

    def _bench_compute(self, progress_cb: Any) -> dict[str, Any]:
        """FNO 학습을 실행하고 결과를 반환한다 (동기 워커 — 상태 미변경)."""
        return bench.train_operator(
            self._bench_dataset,
            backend=self.state.nt_bench_backend or "neuralop",
            epochs=int(self.state.nt_bench_epochs or 60),
            modes=int(self.state.nt_bench_modes or 12),
            width=int(self.state.nt_bench_width or 32),
            progress_cb=progress_cb,
        )

    def _apply_bench_result(self, result: dict[str, Any]) -> str:
        """학습 결과를 상태/차트에 반영한다 (loop 스레드에서만 호출)."""
        self._bench_result = result
        summary = (
            f"{result['operator'].upper()} [{result.get('backend', 'builtin')}] · "
            f"{result['n_train']}train/{result['n_test']}test · "
            f"loss {result['final_loss']:.3g} · test RMSE {result['test_rmse']:.3g} "
            f"(nRMSE {result['test_rel_l2']:.3g}) · "
            f"학습 {result['train_time_s']:.1f}s · 추론 {result['latency_ms']:.2f}ms"
        )
        with self.state:
            self.state.nt_bench_trained = True
            self.state.nt_bench_train_summary = summary
        self._show_chart(
            self._plot_bench_loss(result),
            title=f"FNO 학습 손실 — {self._bench_dataset.get('source', '')}",
        )
        return summary

    def bench_train(self) -> None:
        """FNO 연산자 학습 (동기 — 테스트/폴백 경로)."""
        if self._bench_dataset is None:
            self._fail("데이터셋 없음", RuntimeError("먼저 데이터셋을 생성/로드하세요."))
            return
        try:
            summary = self._apply_bench_result(self._bench_compute(None))
            self._set_status(f"연산자 학습 완료 — {summary}")
        except Exception as exc:  # noqa: BLE001
            self._fail("연산자 학습 실패", exc)

    async def _bench_train_async(self) -> None:
        """FNO 학습 (라이브 진행 — epoch별 진행바/손실 스파크라인 스트리밍)."""
        import asyncio

        if self._bench_dataset is None:
            self._fail("데이터셋 없음", RuntimeError("먼저 데이터셋을 생성/로드하세요."))
            return
        loop = asyncio.get_running_loop()
        self._ensure_progress_monitor()
        losses: list[float] = []

        def progress_cb(epoch: int, total: int, loss: float) -> None:
            # 워커 스레드 → 큐(스레드 안전). 모니터가 loop 스레드에서 적용/push.
            losses.append(float(loss))
            self._progress_queue.put_nowait(
                {
                    "nt_progress": 100.0 * epoch / max(1, total),
                    "nt_bench_epoch": int(epoch),
                    "nt_bench_epochs_total": int(total),
                    "nt_bench_loss": float(loss),
                    "nt_bench_loss_series": list(losses),
                }
            )

        with self.state:
            self.state.nt_busy = True
            self.state.nt_error = ""
            self.state.nt_bench_training = True
            self.state.nt_bench_loss_series = []
            self.state.nt_bench_epoch = 0
            self.state.nt_progress = 0.0
            self.state.nt_status = "신경 연산자(FNO) 학습 중…"
        self.state.flush()
        try:
            result = await loop.run_in_executor(
                self._get_executor(), lambda: self._bench_compute(progress_cb)
            )
        except Exception as exc:  # noqa: BLE001
            with self.state:
                self.state.nt_bench_training = False
                self.state.nt_progress = -1.0
            self._fail("연산자 학습 실패", exc)
            return
        summary = self._apply_bench_result(result)  # loop 스레드
        # 종료 플래그는 즉시(loop 스레드), 진행률 리셋만 큐 마지막에 넣어
        # 늦게 도착한 epoch progress msg 보다 뒤에 -1 이 적용되게 한다.
        self._progress_queue.put_nowait({"nt_progress": -1.0})
        with self.state:
            self.state.nt_bench_training = False
            self.state.nt_busy = False
        self._set_status(f"연산자 학습 완료 — {summary}")

    def _ensure_progress_monitor(self) -> None:
        """라이브 진행 큐 모니터 태스크를 1회 기동한다 (loop 스레드에서 호출)."""
        if self._progress_monitor is None:
            from trame_server.utils.asynchronous import create_state_queue_monitor_task

            self._progress_monitor = create_state_queue_monitor_task(
                self.server, self._progress_queue, delay=0.12
            )

    def bench_evaluate(self) -> None:
        """선택 샘플의 참값 vs FNO 예측을 차트로 비교한다."""
        if self._bench_result is None or self._bench_dataset is None:
            self._fail("모델 없음", RuntimeError("먼저 연산자를 학습하세요."))
            return
        try:
            ev = bench.evaluate_sample(
                self._bench_result["model"],
                self._bench_dataset,
                int(self.state.nt_bench_sample or 0),
            )
            self._show_chart(
                self._plot_bench_eval(ev),
                title=(
                    f"샘플 #{ev['index']} 예측 — RMSE {ev['rmse']:.3g}, "
                    f"{ev['latency_ms']:.2f}ms"
                ),
            )
            self._set_status(
                f"샘플 #{ev['index']} 평가 완료: RMSE {ev['rmse']:.3g}, {ev['latency_ms']:.2f}ms"
            )
        except Exception as exc:  # noqa: BLE001
            self._fail("샘플 평가 실패", exc)

    def _twin_param_values(self) -> tuple[list[float], str]:
        """현재 ③Twin 슬라이더의 입력 벡터와 사람이 읽을 라벨을 만든다.

        문제 유형 A 면 시간 t 하나, 문제 유형 B(케이스 세트)면 운전조건 k 개.
        """
        if self.state.nt_case_mode:
            values = [float(v) for v in (self.state.nt_twin_params or [])]
            names = [str(n) for n in (self.state.nt_param_names or [])]
            point = (
                ", ".join(f"{n}={v:.4g}" for n, v in zip(names, values))
                or "파라미터 없음"
            )
            return values, point
        values = [float(self.state.nt_twin_param or 0.0)]
        return values, f"t={values[0]:.4g}"

    def predict(self) -> None:
        """현재 입력 지점에서 필드를 예측하고 3D 뷰어에 표시한다.

        입력은 문제 유형 A 면 시간 t 하나, 문제 유형 B(케이스 세트)면 파라미터
        슬라이더 k 개의 운전조건 벡터다.
        """
        if self.engine is None or self.dataset is None:
            self._fail("트윈 없음", RuntimeError("먼저 트윈을 학습하세요."))
            return
        try:
            values, point = self._twin_param_values()
            # 형상 가변 케이스에는 "학습 격자" 가 없다(케이스마다 자기 격자를 가짐)
            # → 지금 보고 있는 형상 위에 예측한다. 신경장이라 좌표만 있으면 된다.
            if self.engine.training_metadata.get("varying_mesh"):
                predicted, names = service.predict_to_mesh(
                    self.engine, values, self.dataset
                )
                self._swap_view_dataset(
                    predicted,
                    status=f"예측 완료: {point} → 보고 있는 형상 위에 표시",
                    prefer=names[0] if names else "",
                )
                return
            prediction = service.predict_twin(self.engine, values)
            # 다중 출력(Physics AI) 이면 필드별로 잘라 각각 twin_<name> 으로 붙인다.
            parts = service.split_multi_prediction(self.engine, prediction)
            if parts:
                names = [
                    service.attach_prediction(
                        self.dataset, segment, field_name=f"twin_{display}"
                    )
                    for display, segment in parts
                ]
                shown = names[0]
                label = ", ".join(f"'{n}'" for n in names)
            else:
                shown = service.attach_prediction(self.dataset, prediction)
                label = f"'{shown}'"
            self._refresh_fields(prefer=shown)
            self._render(reset_camera=False)
            self._set_status(f"예측 완료: {point} → {label} 3D 표시")
        except Exception as exc:  # noqa: BLE001
            self._fail("예측 실패", exc)

    def predict_on_mesh(self) -> None:
        """선택한 파일의 메쉬 좌표에서 예측하고 그 메쉬를 뷰어로 전환한다 (M3).

        Physics AI 전용 — 신경장은 학습 격자에 묶이지 않아 임의 좌표에서
        평가된다. 학습 상태(engine/케이스)는 그대로 두고 뷰어 대상만 바꾼다.
        """
        from pathlib import Path

        path = (self.state.nt_path or "").strip()
        if self.engine is None:
            self._fail("트윈 없음", RuntimeError("먼저 트윈을 학습하세요."))
            return
        if not path:
            self._fail("경로 필요", ValueError("예측할 메쉬 파일을 선택하세요."))
            return
        try:
            target = service.load_dataset(path)
            values, point = self._twin_param_values()
            dataset, attached = service.predict_to_mesh(self.engine, values, target)
            if self._origin_dataset is None:
                self._origin_dataset = self.dataset
            label = Path(path).name
            self._swap_view_dataset(
                dataset,
                prefer=attached[0] if attached else "",
                status=f"'{label}' 격자에 예측 완료: {point} → {', '.join(attached)}",
            )
            with self.state:
                self.state.nt_predict_mesh_name = label
        except Exception as exc:  # noqa: BLE001
            self._fail("다른 격자 예측 실패", exc)

    def restore_training_mesh(self) -> None:
        """예측 격자에서 원래 학습 격자로 뷰어를 되돌린다."""
        if self._origin_dataset is None:
            return
        dataset = self._origin_dataset
        self._origin_dataset = None
        self._swap_view_dataset(dataset, status="학습 격자로 복귀했습니다.")
        with self.state:
            self.state.nt_predict_mesh_name = ""

    def _swap_view_dataset(
        self, dataset: Any, *, status: str, prefer: str = ""
    ) -> None:
        """뷰어 대상 데이터셋만 교체한다 — 학습 상태(engine/케이스)는 보존.

        :meth:`_set_dataset` 과 달리 모델/케이스/POD 상태를 리셋하지 않는다.
        """
        self.dataset = dataset
        info = service.dataset_info(dataset)
        n_steps = int(info["time_steps"])
        with self.state:
            self.state.nt_error = ""
            self.state.nt_info_points = info["points"]
            self.state.nt_info_cells = info["cells"]
            self.state.nt_info_steps = n_steps
            self.state.nt_info_fields = ", ".join(info["fields"]) or "-"
            self.state.nt_nsteps = max(1, n_steps)
            self.state.nt_has_timesteps = n_steps > 1
            self.state.nt_timestep = 0
        self._refresh_fields(prefer=prefer)
        self._render(reset_camera=True)
        self._set_status(status)

    def reset_view(self) -> None:
        """카메라를 기본 뷰로 리셋한다."""
        self._render(reset_camera=True)

    # ------------------------------------------------------------------
    # Export callbacks (④)
    # ------------------------------------------------------------------

    def _export_path(self, filename: str) -> str:
        import os

        base = (self.state.nt_export_dir or "").strip() or os.path.join(
            os.path.expanduser("~"), "naviertwin-web"
        )
        return os.path.join(base, filename)

    def _snapshot(self) -> Any:
        """현재 timestep 을 단일 스냅샷 데이터셋으로 materialize 한다 (export 용)."""
        return service.snapshot_dataset(self.dataset, int(self.state.nt_timestep or 0))

    def _export_done(self, path: str, label: str) -> None:
        with self.state:
            self.state.nt_export_last = path
        self._set_status(f"{label} 저장 완료: {path}")

    def export_screenshot(self) -> None:
        """현재 3D 뷰를 PNG 로 저장한다."""
        if self.plotter is None:
            self._fail("뷰어 없음", RuntimeError("3D 뷰어가 준비되지 않았습니다."))
            return
        try:
            path = self._export_path("naviertwin_view.png")
            import os

            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.plotter.screenshot(path)
            self._export_done(path, "스크린샷")
        except Exception as exc:  # noqa: BLE001
            self._fail("스크린샷 저장 실패", exc)

    def export_csv(self) -> None:
        if self.dataset is None:
            self._fail("데이터 없음", RuntimeError("먼저 데이터를 로드하세요."))
            return
        try:
            path = service.export_field_csv(self._snapshot(), self._export_path("fields.csv"))
            self._export_done(path, "필드 CSV")
        except Exception as exc:  # noqa: BLE001
            self._fail("CSV 저장 실패", exc)

    def export_vtk(self) -> None:
        if self.dataset is None:
            self._fail("데이터 없음", RuntimeError("먼저 데이터를 로드하세요."))
            return
        try:
            path = service.export_vtk(self._snapshot(), self._export_path("mesh.vtk"))
            self._export_done(path, "메쉬 VTK")
        except Exception as exc:  # noqa: BLE001
            self._fail("VTK 저장 실패", exc)

    def export_project(self) -> None:
        if self.dataset is None:
            self._fail("데이터 없음", RuntimeError("먼저 데이터를 로드하세요."))
            return
        try:
            paths = service.save_project(
                self._snapshot(), self._export_path("project.ntwin"), engine=self.engine
            )
            label = ".ntwin 프로젝트" + (" + engine" if "engine" in paths else "")
            self._export_done(paths["project"], label)
        except Exception as exc:  # noqa: BLE001
            self._fail("프로젝트 저장 실패", exc)

    def export_engine(self) -> None:
        if self.engine is None:
            self._fail("모델 없음", RuntimeError("먼저 ②Model 에서 학습하세요."))
            return
        try:
            path = service.save_engine(self.engine, self._export_path("engine.pkl"))
            self._export_done(path, "TwinEngine")
        except Exception as exc:  # noqa: BLE001
            self._fail("엔진 저장 실패", exc)

    def export_physicsnemo(self) -> None:
        """학습된 Physics AI 모델을 표준 PhysicsNeMo Module 체크포인트로 저장한다."""
        if self.engine is None:
            self._fail("모델 없음", RuntimeError("먼저 ②Model 에서 PhysicsNeMo를 학습하세요."))
            return
        try:
            path = service.export_physicsnemo_module(
                self.engine, self._export_path("physicsnemo_module.pt")
            )
            self._export_done(path, "PhysicsNeMo Module")
        except Exception as exc:  # noqa: BLE001
            self._fail("PhysicsNeMo Module 저장 실패", exc)

    def export_report(self) -> None:
        if self.dataset is None:
            self._fail("데이터 없음", RuntimeError("먼저 데이터를 로드하세요."))
            return
        try:
            metrics: dict[str, Any] = {}
            if self.state.nt_pod_energy:
                metrics["pod_cumulative_energy_pct"] = float(self.state.nt_pod_energy[-1])
            model_info = {}
            if self.state.nt_model_ready:
                model_info["model"] = self.state.nt_model_summary
            path = service.export_report(
                self._snapshot(),
                self._export_path("report.html"),
                summary=self.state.nt_status,
                model_info=model_info,
                metrics=metrics,
            )
            self._export_done(path, "HTML 보고서")
        except Exception as exc:  # noqa: BLE001
            self._fail("보고서 저장 실패", exc)

    # ------------------------------------------------------------------
    # View state
    # ------------------------------------------------------------------

    def _on_view_state_change(self, **_kwargs: Any) -> None:
        if self.dataset is not None:
            self._render(reset_camera=False)

    def _set_dataset(self, dataset: Any, *, status: str) -> None:
        """단일 데이터셋(문제 유형 A)을 적재하고 파생 상태를 전부 리셋한다.

        케이스 세트 로드는 :meth:`_set_case_set` 이 이 메서드를 먼저 호출해
        리셋한 뒤 케이스 상태를 다시 채운다 — 순서 의존적이다.
        """
        self.dataset = dataset
        self.case_datasets = None
        self.case_params = None
        self.case_param_names = []
        self._origin_dataset = None
        self.reducer = None
        self.engine = None
        info = service.dataset_info(dataset)
        n_steps = int(info["time_steps"])
        with self.state:
            self.state.nt_has_dataset = True
            self.state.nt_error = ""
            self.state.nt_info_points = info["points"]
            self.state.nt_info_cells = info["cells"]
            self.state.nt_info_steps = n_steps
            self.state.nt_info_fields = ", ".join(info["fields"]) or "-"
            self.state.nt_info_source = info["source"]
            self.state.nt_nsteps = n_steps
            self.state.nt_has_timesteps = n_steps > 1
            self.state.nt_timestep = 0
            self.state.nt_case_mode = False
            self.state.nt_case_count = 0
            self.state.nt_case_names = []
            self.state.nt_params_source = ""
            self.state.nt_param_names = []
            self.state.nt_case_index = 0
            self.state.nt_case_labels = []
            self.state.nt_case_resampled = False
            self.state.nt_case_grid_summary = ""
            self.state.nt_coarsen_summary = ""
            self.state.nt_twin_params = []
            self.state.nt_twin_mins = []
            self.state.nt_twin_maxs = []
            self.state.nt_twin_steps = []
            self.state.nt_predict_mesh_name = ""
            self.state.nt_analysis_done = False
            self.state.nt_pod_done = False
            self.state.nt_pod_summary = ""
            self.state.nt_pod_energy = []
            self.state.nt_pod_mode = 0
            self.state.nt_pod_max_mode = 0
            self.state.nt_model_ready = False
            self.state.nt_model_summary = ""
            self.state.nt_physics_ready = False
            self.state.nt_dmd_ready = False
            self.state.nt_dmd_fit_error = 0.0
            self.state.nt_dmd_summary = ""
            base_fields = [n for n in info["fields"] if not self._is_derived_field(n)]
            default_field = render.preferred_field(base_fields)
            self.state.nt_train_field_choices = base_fields
            self.state.nt_train_field = default_field
            self.state.nt_train_fields = [default_field] if default_field else []
            self.state.nt_train_input_fields = []
            try:
                self.state.nt_method_hint = service.recommend_method(dataset)["reason"]
                self.state.nt_strategy_status = service.strategy_status(dataset)
            except Exception:  # noqa: BLE001 — 추천은 부가 정보, 로드를 막지 않는다
                self.state.nt_method_hint = ""
                self.state.nt_strategy_status = {}
            self.state.nt_twin_ready = False
            self.state.nt_twin_summary = ""
            self.state.nt_fft_summary = ""
            self.state.nt_fft_field = ""
            self.state.nt_fft_probe = False
            self.state.nt_fft_point = 0
            self.state.nt_chart_dialog = False
            self.state.nt_compare_dialog = False
            self.state.nt_compare_rows = []
            self.state.nt_compare_summary = ""
        self._pod_result = None
        self._refresh_fields()
        self._update_coarsen_preview()
        self._render(reset_camera=True)
        self._set_status(status)

    def _update_coarsen_preview(self, **_kwargs: Any) -> None:
        """현재 해상도 값이면 결과가 몇 점이 되는지 미리 보여준다.

        재샘플은 되돌릴 수 없으므로(원본 교체) 적용 전에 대가를 보여야 한다.
        격자 치수만 계산하는 저렴한 추정이라 슬라이더를 움직일 때마다 호출해도
        된다.
        """
        if self.dataset is None or self.state.nt_case_mode:
            with self.state:
                self.state.nt_coarsen_preview = ""
                self.state.nt_coarsen_increases = False
            return
        increases = False
        try:
            estimate = service.estimate_coarsen(
                self.dataset, int(self.state.nt_coarsen_resolution or 48)
            )
            preview = f"→ {estimate['summary']}"
            # 원본보다 촘촘해지면 "낮추기"가 아니라 오히려 손해다 — 경고색으로.
            increases = float(estimate["ratio"]) < 1.0
        except Exception:  # noqa: BLE001 — 미리보기는 부가 정보, UI 를 막지 않는다
            preview = ""
        with self.state:
            self.state.nt_coarsen_preview = preview
            self.state.nt_coarsen_increases = increases

    def _refresh_fields(self, prefer: str = "") -> None:
        names = render.available_fields(self.dataset) if self.dataset is not None else []
        current = prefer or self.state.nt_field
        if current not in names:
            current = render.preferred_field(names)
        with self.state:
            self.state.nt_fields = names
            self.state.nt_field = current

    @staticmethod
    def _is_derived_field(name: str) -> bool:
        """분석/예측/모드로 생성된 파생 field 인지 판정한다 (service SSOT 위임)."""
        return service.is_derived_field(name)

    def _base_field(self) -> str:
        """⑤부가 분석(POD/FFT 진단) 대상 field — 3D 뷰어의 nt_field 를 따라간다.

        진단 도구는 "지금 보고 있는 필드를 더 들여다본다"는 성격이라 뷰어와
        같은 field 를 쓰는 편이 자연스럽다. ②Model 학습 대상은 이것과 별개인
        :meth:`_training_field` 를 쓴다 — 근거: 사용자 지적("트윈 학습할 때
        입력/출력 설정이 없냐") → 뷰어 컬러링용 필드에 학습 대상이 몰래
        종속돼 있던 문제.
        """
        names = render.available_fields(self.dataset) if self.dataset is not None else []
        base = [n for n in names if not self._is_derived_field(n)]
        field = self.state.nt_field or ""
        if field and not self._is_derived_field(field):
            return field
        return render.preferred_field(base) if base else field

    def _training_field(self) -> str:
        """②Model 학습 대상(출력) field — 패널 안의 명시적 선택을 우선한다.

        ``nt_train_field`` 는 데이터 로드 시 기본값이 채워지고 ②Model 의
        "출력 필드" 드롭다운으로 사용자가 바꿀 수 있다 — 3D 뷰어의 nt_field
        (색상 표시용) 와 완전히 독립적이다.
        """
        choices = list(self.state.nt_train_field_choices or [])
        chosen = str(self.state.nt_train_field or "")
        if chosen and chosen in choices:
            return chosen
        return self._base_field()

    def _training_fields(self) -> list[str]:
        """Physics AI 학습 대상(출력) 필드 목록 — 다중 선택을 지원한다.

        ``nt_train_fields`` 의 유효한 선택만 남기고, 비어 있으면 단일 선택
        (:meth:`_training_field`)으로 폴백한다.
        """
        choices = list(self.state.nt_train_field_choices or [])
        selected = [str(f) for f in (self.state.nt_train_fields or []) if str(f) in choices]
        return selected or [self._training_field()]

    def _training_input_fields(self) -> list[str]:
        """Physics AI 의 입력 field 목록 — 출력으로 쓰는 field 는 제외한다.

        같은 field 를 입력이자 출력으로 주면 항등 매핑을 학습하게 되므로 코어가
        거부한다. UI 실수를 여기서 미리 걸러 명확한 상태를 만든다.
        """
        choices = list(self.state.nt_train_field_choices or [])
        outputs = set(self._training_fields())
        return [
            str(f)
            for f in (self.state.nt_train_input_fields or [])
            if str(f) in choices and str(f) not in outputs
        ]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _ensure_plotter(self) -> Any:
        if self.plotter is None:
            import pyvista as pv

            pv.OFF_SCREEN = True
            self.plotter = pv.Plotter(off_screen=True)
            try:
                self.plotter.set_background(
                    theme.VIEWER_BG_BOTTOM, top=theme.VIEWER_BG_TOP
                )
            except Exception:  # noqa: BLE001
                self.plotter.background_color = theme.VIEWER_BG_BOTTOM
            try:
                self.plotter.show_axes()
            except Exception:  # noqa: BLE001
                pass
        return self.plotter

    def _flush_render(self) -> None:
        """비동기 워커가 표시한 보류 렌더를 메인 스레드에서 수행한다 (GL 안전)."""
        if self._render_pending:
            self._render_pending = False
            reset = self._reset_camera_pending
            self._reset_camera_pending = False
            self._render(reset_camera=reset)

    def _render(self, reset_camera: bool = False) -> None:
        # executor(워커 스레드) 실행 중에는 렌더를 메인 스레드로 미룬다.
        if self._defer_render:
            self._render_pending = True
            self._reset_camera_pending = self._reset_camera_pending or reset_camera
            return
        if self.dataset is None or self.plotter is None:
            return
        try:
            mesh, scalar = render.prepare_render_mesh(
                self.dataset,
                self.state.nt_field or "",
                int(self.state.nt_timestep or 0),
            )
        except Exception as exc:  # noqa: BLE001
            self._fail("렌더 준비 실패", exc)
            return

        camera = None
        if not reset_camera:
            try:
                camera = self.plotter.camera_position
            except Exception:  # noqa: BLE001
                camera = None

        self.plotter.clear()
        kwargs: dict[str, Any] = {
            "show_edges": bool(self.state.nt_show_edges),
            "name": "nt_mesh",
        }
        if scalar:
            kwargs.update(
                scalars=scalar,
                cmap=self.state.nt_cmap,
                nan_color="#8b949e",
                # 넓은 뷰포트의 우측 여백을 세로 컬러바로 채워 화면 균형을 맞춘다.
                scalar_bar_args={
                    "vertical": True,
                    "position_x": 0.88,
                    "position_y": 0.2,
                    "height": 0.6,
                    "width": 0.05,
                    "fmt": "%.3g",
                    "n_labels": 5,
                    "title_font_size": 15,
                    "label_font_size": 12,
                },
            )
        else:
            kwargs.update(color=render.SOLID_COLOR)
        self.plotter.add_mesh(mesh, **kwargs)
        try:
            self.plotter.show_axes()
        except Exception:  # noqa: BLE001
            pass

        if reset_camera or camera is None:
            try:
                if render.mesh_is_flat(mesh):
                    self.plotter.view_xy()
                    self.plotter.reset_camera()
                    # 2D 필드는 뷰포트 세로를 채우도록 확대해 여백을 줄인다.
                    try:
                        self.plotter.camera.zoom(1.35)
                    except Exception:  # noqa: BLE001
                        pass
                else:
                    self.plotter.view_isometric()
                    self.plotter.reset_camera()
            except Exception:  # noqa: BLE001
                pass
        else:
            try:
                self.plotter.camera_position = camera
            except Exception:  # noqa: BLE001
                pass

        update = getattr(self.ctrl, "view_update", None)
        if callable(update):
            update()

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _set_status(self, message: str, *, toast: bool = True) -> None:
        with self.state:
            self.state.nt_status = message
            self.state.nt_error = ""
            self.state.nt_busy = False
            if toast:
                self.state.nt_toast = message
                self.state.nt_toast_color = "success"
                self.state.nt_toast_icon = "mdi-check-circle"
                self.state.nt_toast_show = True

    def _fail(self, title: str, exc: Exception) -> None:
        log.warning("%s: %s", title, exc)
        detail = str(exc) or title
        with self.state:
            # footer 는 status(제목) + error(상세) 두 span 을 나란히 표시하므로
            # error 에 제목을 반복하지 않는다 ("데이터 로드 실패데이터 로드 실패: …" 방지).
            self.state.nt_error = detail
            self.state.nt_status = title
            self.state.nt_busy = False
            self.state.nt_toast = f"{title}: {detail}"
            self.state.nt_toast_color = "error"
            self.state.nt_toast_icon = "mdi-alert-circle"
            self.state.nt_toast_show = True

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _tip(
        self, text: str, *, warn: bool = False, v_show_expr: str = ""
    ) -> None:
        """긴 설명을 패널에서 걷어내 호버 툴팁으로 옮긴다.

        해설이 패널에 상주하면 정작 봐야 할 컨트롤과 상태값이 묻힌다. 설명은
        회색 ⓘ, 경고는 주황 ⚠ 로 아이콘 색만 다르게 해서 화면을 비우면서도
        "여기 주의할 게 있다"는 신호는 남긴다.

        아이콘은 버튼 **옆**에 별도로 둔다 — 비활성(disabled) 버튼에는 hover
        이벤트가 걸리지 않아 툴팁이 뜨지 않기 때문이다. 버튼이 비활성인 순간이
        바로 "왜 안 되는지" 설명이 필요한 순간이므로 이게 중요하다.

        Args:
            text: 툴팁 내용. ``{{ ... }}`` 바인딩도 쓸 수 있다.
            warn: 경고면 True — 주황 ⚠ 아이콘이 된다.
            v_show_expr: 조건부로 띄울 때의 vue 표현식 (빈 문자열이면 항상).
        """
        from trame.widgets import html
        from trame.widgets import vuetify3 as v3

        icon_kwargs: dict[str, Any] = {}
        if v_show_expr:
            icon_kwargs["v_show"] = (v_show_expr,)
        with v3.VTooltip(location="bottom", max_width=340, open_delay=80):
            with v3.Template(v_slot_activator="{ props }"):
                v3.VIcon(
                    "mdi-alert-circle-outline" if warn else "mdi-information-outline",
                    v_bind="props",
                    size="x-small",
                    color="warning" if warn else "grey",
                    classes="nt-tip ml-1",
                    **icon_kwargs,
                )
            html.Span(text, classes="text-caption")

    def _tip_row(self, label: str, text: str, *, warn: bool = False) -> None:
        """라벨 + ⓘ 를 한 줄로 (섹션 제목 옆에 붙이는 경우)."""
        from trame.widgets import html

        with html.Div(classes="d-flex align-center"):
            html.Span(label, classes="text-caption text-disabled")
            self._tip(text, warn=warn)

    def build_ui(self) -> Any:
        """trame UI 레이아웃을 구성한다 (PyVista Plotter + vuetify3 컨트롤)."""
        from pyvista.trame.ui import plotter_ui
        from trame.ui.vuetify3 import SinglePageWithDrawerLayout
        from trame.widgets import client, html
        from trame.widgets import vuetify3 as v3

        self._ensure_plotter()

        # 다크 테마 주입 (VAppLayout 이 state.trame__vuetify3_config 로 읽음)
        self.server.state.trame__vuetify3_config = theme.vuetify_config()

        with SinglePageWithDrawerLayout(
            self.server, vuetify_config=theme.vuetify_config()
        ) as layout:
            layout.title.set_text("")  # 브랜드를 toolbar 커스텀 span 으로 대체
            with layout.toolbar:
                v3.VIcon("mdi-waves", color="primary", classes="ml-1 mr-2")
                html.Span("NavierTwin", classes="nt-brand text-h6")
                html.Span(
                    "CFD Digital Twin",
                    classes="text-caption text-disabled ml-2 d-none d-md-flex",
                )
                v3.VSpacer()
                v3.VSelect(
                    v_model=("nt_field",),
                    items=("nt_fields",),
                    label="Field",
                    density="compact",
                    hide_details=True,
                    style="max-width: 180px",
                    disabled=("!nt_has_dataset",),
                )
                v3.VSelect(
                    v_model=("nt_cmap",),
                    items=("nt_cmaps",),
                    label="Colormap",
                    density="compact",
                    hide_details=True,
                    style="max-width: 150px",
                    classes="ml-2",
                    disabled=("!nt_has_dataset",),
                )
                v3.VCheckbox(
                    v_model=("nt_show_edges",),
                    label="Edges",
                    density="compact",
                    hide_details=True,
                    classes="ml-2",
                    disabled=("!nt_has_dataset",),
                )
                v3.VBtn(
                    "Reset View",
                    click=self.ctrl.nt_reset_view,
                    variant="tonal",
                    size="small",
                    classes="ml-2",
                    disabled=("!nt_has_dataset",),
                )
                # 결정형(nt_progress>=0)/비결정형(nt_busy && nt_progress<0) 겸용 진행바
                v3.VProgressLinear(
                    active=("nt_busy",),
                    indeterminate=("nt_busy && nt_progress < 0",),
                    model_value=("nt_progress < 0 ? 0 : nt_progress",),
                    absolute=True,
                    location="bottom",
                    color="primary",
                    height=3,
                )

            with layout.drawer as drawer:
                drawer.width = 344
                drawer.classes = "nt-drawer"
                self._build_pipeline_strip(v3, html)
                self._build_drawer(v3, html)

            with layout.content:
                # 전역 커스텀 CSS — Vue 는 <style> 를 템플릿에서 제거하므로 trame
                # 전용 <trame-style>(client.Style) 로 <head> 에 런타임 주입한다.
                client.Style(theme.CUSTOM_CSS)
                with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
                    view = plotter_ui(self.plotter, default_server_rendering=True)
                    self.ctrl.view_update = view.update

            # 상태 / 오류 표시줄
            with layout.footer as footer:
                footer.clear()
                footer.style = "display:flex; align-items:center; gap:12px;"
                html.Span("{{ nt_status }}", classes="text-caption")
                v3.VSpacer()
                html.Span(
                    "{{ nt_error }}",
                    classes="text-caption text-error",
                    v_show=("nt_error",),
                )
                html.Span(f"v{__version__}", classes="text-caption text-disabled")

            # 차트 모달 (FFT/PSD, POD 에너지)
            with v3.VDialog(v_model=("nt_chart_dialog",), max_width="820"):
                with v3.VCard():
                    with v3.VCardTitle(classes="text-subtitle-1"):
                        html.Span("{{ nt_chart_title }}")
                    with v3.VCardText(classes="text-center"):
                        html.Img(
                            src=("nt_chart_img",),
                            style="max-width:100%; height:auto;",
                        )
                    with v3.VCardActions():
                        v3.VSpacer()
                        v3.VBtn("닫기", click="nt_chart_dialog = false", variant="text")

            # 모델 비교 결과 표 모달
            with v3.VDialog(v_model=("nt_compare_dialog",), max_width="780"):
                with v3.VCard():
                    with v3.VCardTitle(classes="text-subtitle-1"):
                        html.Span("모델 비교 — RMSE 오름차순")
                    with v3.VCardText():
                        html.Div("{{ nt_compare_summary }}", classes="text-caption mb-2")
                        v3.VDataTable(
                            headers=("nt_compare_headers",),
                            items=("nt_compare_rows",),
                            item_value="combo",
                            density="compact",
                            items_per_page=-1,
                        )
                    with v3.VCardActions():
                        v3.VSpacer()
                        v3.VBtn("닫기", click="nt_compare_dialog = false", variant="text")

            # 파일 브라우저 모달 (서버측 탐색기)
            with v3.VDialog(v_model=("nt_fb_dialog",), max_width="640"):
                with v3.VCard():
                    with v3.VCardTitle(classes="text-subtitle-1 d-flex align-center"):
                        v3.VIcon("mdi-folder-open", classes="mr-2", color="primary")
                        html.Span(
                            "{{ nt_fb_mode === 'caseset' ? '케이스 세트 폴더 선택' "
                            ": (nt_fb_mode === 'predict_mesh' ? '예측할 격자 파일 선택' "
                            ": '파일 · 폴더 열기') }}"
                        )
                    with v3.VCardSubtitle(classes="text-caption text-truncate pb-2"):
                        html.Span("{{ nt_fb_cwd }}")
                    html.Div(
                        "케이스 파일들이 들어있는 폴더로 이동한 뒤 아래 버튼을 "
                        "누르세요 (파일 클릭 아님).",
                        v_if="nt_fb_mode === 'caseset'",
                        classes="text-caption text-info px-4 pb-2",
                    )
                    # 공통 격자 해상도는 로드 시점에 정해진다 — 케이스 메쉬가 서로
                    # 다르면(형상 가변) 여기서 정한 격자로 재샘플되기 때문이다.
                    with html.Div(v_if="nt_fb_mode === 'caseset'", classes="px-4 pb-2"):
                        self._tip_row(
                            "공통 격자 해상도: 긴 축 {{ nt_case_resolution }}분할",
                            "케이스 메쉬가 서로 다를 때만 쓰입니다 — 모두 같은 "
                            "메쉬면 재샘플 없이 원본 그대로 씁니다. 형상이 다른 "
                            "케이스는 이 격자로 보간되므로, 형상 경계를 살리려면 "
                            "충분히 촘촘해야 합니다.",
                        )
                        v3.VSlider(
                            v_model=("nt_case_resolution",),
                            min=8,
                            max=96,
                            step=4,
                            thumb_label=True,
                            density="compact",
                            hide_details=True,
                        )
                    html.Div(
                        "예측 결과를 올릴 메쉬 파일을 클릭하세요 — 그 격자의 "
                        "좌표에서 트윈을 평가합니다.",
                        v_if="nt_fb_mode === 'predict_mesh'",
                        classes="text-caption text-info px-4 pb-2",
                    )
                    v3.VDivider()
                    with v3.VCardText(
                        classes="pa-0",
                        style="max-height:52vh; overflow-y:auto;",
                    ):
                        with v3.VList(density="compact", nav=True):
                            v3.VListItem(
                                v_if="nt_fb_parent",
                                title="..",
                                prepend_icon="mdi-arrow-up-left",
                                click=(self.ctrl.nt_fb_navigate, "[nt_fb_parent]"),
                            )
                            with html.Template(
                                v_for="entry in nt_fb_entries",
                                key="entry.path",
                            ):
                                v3.VListItem(
                                    v_if="entry.is_dir",
                                    title=("entry.name",),
                                    prepend_icon="mdi-folder",
                                    click=(self.ctrl.nt_fb_navigate, "[entry.path]"),
                                )
                                v3.VListItem(
                                    v_if="!entry.is_dir",
                                    title=("entry.name",),
                                    prepend_icon="mdi-file-outline",
                                    click=(self.ctrl.nt_fb_pick, "[entry.path]"),
                                )
                        html.Div(
                            "(이 폴더에는 하위 폴더나 로드 가능한 파일이 없습니다)",
                            v_if="nt_fb_entries.length === 0",
                            classes="text-caption text-disabled pa-4 text-center",
                        )
                    v3.VDivider()
                    with v3.VCardActions():
                        # 모드에 따라 폴더 액션이 달라진다: single = 폴더 자체가
                        # 케이스 1개(OpenFOAM), caseset = 폴더 안 파일들이 케이스 N개.
                        v3.VBtn(
                            "현재 폴더 로드",
                            click=self.ctrl.nt_fb_load_cwd,
                            variant="tonal",
                            color="primary",
                            prepend_icon="mdi-folder-download-outline",
                            v_if="nt_fb_mode === 'single'",
                        )
                        v3.VBtn(
                            "이 폴더를 케이스 세트로 로드",
                            click=self.ctrl.nt_fb_load_case_set,
                            variant="tonal",
                            color="primary",
                            prepend_icon="mdi-folder-multiple-outline",
                            v_if="nt_fb_mode === 'caseset'",
                        )
                        v3.VSpacer()
                        v3.VBtn("닫기", click="nt_fb_dialog = false", variant="text")

            # 토스트 (완료/오류 알림)
            with v3.VSnackbar(
                v_model=("nt_toast_show",),
                color=("nt_toast_color",),
                timeout=3500,
                location="bottom right",
                variant="elevated",
            ):
                v3.VIcon(("nt_toast_icon",), classes="mr-2")
                html.Span("{{ nt_toast }}")

        self._ui_built = True
        return layout

    def _build_pipeline_strip(self, v3: Any, html: Any) -> None:
        """드로어 상단 워크플로우 진행 칩 — 완료 시 초록/체크, 클릭 시 해당 패널 열기.

        패널 순서를 두 블록(핵심 파이프라인 4단계 vs 보조 도구 2개)으로 나눈 뒤
        칩도 같은 두 줄로 나눠 렌더한다 — 근거: .omc/plans/model-taxonomy-plan.md
        §10 (핵심 트윈 파이프라인 vs 보조 진단/실험 분리).
        """
        core_stages = [
            ("①", "Import", "nt_has_dataset", 0),
            ("②", "Model", "nt_model_ready || !!nt_compare_summary", 1),
            ("③", "Twin", "nt_twin_ready", 2),
            ("④", "Export", "!!nt_export_last", 3),
        ]
        aux_stages = [
            ("⑤", "분석", "nt_analysis_done || nt_pod_done", 4),
            ("⑥", "Lab", "nt_bench_trained", 5),
        ]

        def _chip_row(stages: list[tuple[str, str, str, int]]) -> None:
            with v3.VSheet(color="transparent", classes="d-flex flex-wrap ga-1"):
                for num, name, done, idx in stages:
                    v3.VChip(
                        f"{num} {name}",
                        size="x-small",
                        variant="tonal",
                        color=(f"{done} ? 'success' : 'grey'",),
                        prepend_icon=(
                            f"{done} ? 'mdi-check-circle' : 'mdi-circle-small'",
                        ),
                        click=f"nt_open_panels = [{idx}]",
                        classes=("nt_busy ? 'nt-chip-active' : ''",),
                    )

        with html.Div(classes="px-3 pt-3 pb-1"):
            _chip_row(core_stages)
            html.Div(
                "보조",
                classes="text-caption text-disabled mt-2 mb-1",
            )
            _chip_row(aux_stages)

    def _build_drawer(self, v3: Any, html: Any) -> None:
        with v3.VExpansionPanels(v_model=("nt_open_panels",), multiple=True):
            html.Div(
                "핵심 파이프라인",
                classes="text-overline text-disabled px-1 mt-1 mb-1",
            )
            # 1) Import
            with v3.VExpansionPanel(title="① Import"):
                with v3.VExpansionPanelText():
                    # 데모 카탈로그 — 계열별로 시험 가능한 데이터를 고른다.
                    # 데모 설명은 ⓘ 로: 드롭다운에서 고르는 순간에만 필요하다.
                    with html.Div(classes="d-flex align-center"):
                        v3.VSelect(
                            v_model=("nt_demo_kind",),
                            items=("nt_demo_choices",),
                            label="데모 데이터",
                            density="compact",
                            hide_details=True,
                            classes="flex-grow-1",
                        )
                        self._tip("{{ nt_demo_notes[nt_demo_kind] }}")
                    v3.VBtn(
                        "데모 데이터 로드",
                        click=self.ctrl.nt_load_demo,
                        color="primary",
                        block=True,
                        classes="mt-2",
                        disabled=("nt_busy",),
                        prepend_icon="mdi-flask-outline",
                    )
                    with html.Div(classes="d-flex align-center mt-3"):
                        v3.VBtn(
                            "경로에서 로드",
                            click=self.ctrl.nt_fb_open,
                            variant="tonal",
                            block=True,
                            disabled=("nt_busy",),
                            prepend_icon="mdi-folder-search-outline",
                            classes="flex-grow-1",
                        )
                        self._tip(
                            "탐색기에서 CFD 파일(*.vtk, *.vtu, ...), OpenFOAM 폴더, "
                            "또는 .ntwin 프로젝트를 선택합니다. 같은 폴더의 "
                            "<name>.engine.pkl 이 있으면 트윈도 함께 복원합니다. "
                            "재로드된 프로젝트는 단일 timestep 이라 예측만 "
                            "가능합니다."
                        )
                    # 케이스 세트(문제 유형 B) — 파일 N개 = 운전조건 N개.
                    with html.Div(classes="d-flex align-center mt-3"):
                        v3.VBtn(
                            "케이스 세트 로드 (파라미터 스윕)",
                            click=self.ctrl.nt_fb_open_case_set,
                            variant="tonal",
                            block=True,
                            disabled=("nt_busy",),
                            prepend_icon="mdi-folder-multiple-outline",
                            classes="flex-grow-1",
                        )
                        self._tip(
                            "폴더 하나 = 케이스 여러 개(파일 1개 = 운전조건 1개) + "
                            "입력 파라미터 CSV(행=케이스, 파일명 정렬 순서). 정상 "
                            "해석 결과를 형상/조건별로 학습할 때 씁니다."
                        )
                    with v3.VCard(variant="tonal", classes="mt-3", v_show=("nt_has_dataset",)):
                        with v3.VCardText(classes="text-caption"):
                            html.Div(
                                "케이스: {{ nt_case_count }}개 · 입력 파라미터: "
                                "{{ nt_param_names.join(', ') }} ({{ nt_params_source }})",
                                v_show=("nt_case_mode",),
                                classes="text-info mb-1",
                            )
                            html.Div(
                                "형상 가변 — {{ nt_case_grid_summary }}",
                                v_show=("nt_case_resampled",),
                                classes="text-warning mb-1",
                            )
                            html.Div("Points: {{ nt_info_points }}")
                            html.Div("Cells: {{ nt_info_cells }}")
                            html.Div(
                                "Time steps: {{ nt_info_steps }}",
                                v_show=("!nt_case_mode",),
                            )
                            html.Div("Fields: {{ nt_info_fields }}")
                    # 해상도 낮추기 — 대용량 메쉬는 스냅샷 행렬이 메모리를 넘긴다.
                    with v3.VCard(
                        variant="flat", classes="mt-2", v_show=("nt_has_dataset && !nt_case_mode",)
                    ):
                        with v3.VCardText():
                            self._tip_row(
                                "해상도 낮추기 (대용량 대응)",
                                "성긴 균일 격자로 보간 재샘플합니다 — 학습/POD 는 "
                                "(점 수 × 스텝 수) 행렬을 통째로 메모리에 올리므로 "
                                "거대한 메쉬는 먼저 줄여야 합니다. 손실 압축이라 "
                                "급격한 구배는 뭉개지고, 원본을 교체하므로 되돌릴 "
                                "수 없습니다.",
                                warn=True,
                            )
                            html.Div(
                                "긴 축 기준 {{ nt_coarsen_resolution }}분할",
                                classes="text-caption mb-1",
                            )
                            v3.VSlider(
                                v_model=("nt_coarsen_resolution",),
                                min=8,
                                max=128,
                                step=4,
                                thumb_label=True,
                                density="compact",
                                hide_details=True,
                                disabled=("nt_busy",),
                            )
                            # 재샘플은 원본을 교체하므로, 적용 전에 대가를 보여준다.
                            html.Div(
                                "{{ nt_coarsen_preview }}",
                                v_show=("nt_coarsen_preview",),
                                classes=(
                                    "nt_coarsen_increases "
                                    "? 'text-caption text-warning mt-1' "
                                    ": 'text-caption text-info mt-1'",
                                ),
                            )
                            v3.VBtn(
                                "적용",
                                click=self.ctrl.nt_coarsen,
                                variant="tonal",
                                block=True,
                                classes="mt-2",
                                disabled=("nt_busy",),
                                prepend_icon="mdi-grid-off",
                            )
                            html.Div(
                                "{{ nt_coarsen_summary }}",
                                v_show=("nt_coarsen_summary",),
                                classes="text-caption text-success mt-1",
                            )
                    # 케이스 슬라이더 — 케이스 세트는 시간축이 없어 타임스텝
                    # 슬라이더 대신 이걸로 케이스별 원본 해를 본다.
                    with v3.VCard(variant="flat", classes="mt-2", v_show=("nt_case_mode",)):
                        with v3.VCardText():
                            with html.Div(classes="d-flex align-center mb-1"):
                                html.Span(
                                    "케이스 {{ nt_case_index + 1 }} / "
                                    "{{ nt_case_count }} — "
                                    "{{ nt_case_names[nt_case_index] }}",
                                    classes="text-caption",
                                )
                                self._tip(
                                    "케이스별 원본 해를 봅니다 — ③Twin 예측 결과와 "
                                    "눈으로 비교할 수 있습니다. 케이스를 바꿔도 "
                                    "학습한 모델은 그대로 유지됩니다."
                                )
                            html.Div(
                                "{{ nt_case_labels[nt_case_index] }}",
                                classes="text-caption text-info mb-1",
                            )
                            v3.VSlider(
                                v_model=("nt_case_index",),
                                min=0,
                                max=("nt_case_count - 1",),
                                step=1,
                                hide_details=True,
                                density="compact",
                            )
                    # 타임스텝 슬라이더
                    with v3.VCard(variant="flat", classes="mt-2", v_show=("nt_has_timesteps",)):
                        with v3.VCardText():
                            html.Div(
                                "Timestep: {{ nt_timestep }} / {{ nt_nsteps - 1 }}",
                                classes="text-caption mb-1",
                            )
                            v3.VSlider(
                                v_model=("nt_timestep",),
                                min=0,
                                max=("nt_nsteps - 1",),
                                step=1,
                                hide_details=True,
                                density="compact",
                            )

            # 2) Model — 방식 우선(method-first) 2단 선택.
            # 계열 분류/근거: .omc/plans/model-taxonomy-plan.md
            with v3.VExpansionPanel(title="② Model (트윈 학습)"):
                with v3.VExpansionPanelText():
                    # 입출력 설정 — 무엇을 예측하고(출력) 무엇으로 예측하는지(입력)
                    # 를 명시한다. 이전엔 출력이 3D 뷰어 컬러링용 nt_field 에
                    # 몰래 종속돼 있어 사용자가 알아볼 수 없었다.
                    html.Div("입력 · 출력", classes="text-caption text-disabled mb-1")
                    # ROM(POD)은 스냅샷 행렬 하나 = 출력 1개. Physics AI 는 한
                    # 신경망이 여러 필드를 동시 학습(다중 출력) 가능.
                    v3.VSelect(
                        v_model=("nt_train_field",),
                        items=("nt_train_field_choices",),
                        label="출력 필드 (예측 대상)",
                        density="compact",
                        hide_details=True,
                        v_show=("nt_model_method !== 'physics'",),
                    )
                    v3.VSelect(
                        v_model=("nt_train_fields",),
                        items=("nt_train_field_choices",),
                        label="출력 필드 (복수 선택 가능)",
                        density="compact",
                        hide_details=True,
                        multiple=True,
                        chips=True,
                        closable_chips=True,
                        v_show=("nt_model_method === 'physics'",),
                    )
                    # 입력 field (Physics AI 전용) — ROM 은 파라미터→POD계수 구조라
                    # per-point 입력장을 받을 수 없다.
                    v3.VSelect(
                        v_model=("nt_train_input_fields",),
                        items=("nt_train_field_choices",),
                        label="입력 필드 (선택 — 비우면 시간만)",
                        density="compact",
                        hide_details=True,
                        multiple=True,
                        chips=True,
                        closable_chips=True,
                        classes="mt-2",
                        v_show=("nt_model_method === 'physics'",),
                    )
                    # 입력 파라미터는 "무엇으로 예측하는가" 자체라 상태로 남기고,
                    # 왜 그렇게 정해지는지만 ⓘ 로 접는다.
                    with html.Div(classes="d-flex align-center mt-1 mb-3"):
                        html.Span(
                            "입력 파라미터: 시간(t)",
                            v_show=("!nt_case_mode",),
                            classes="text-caption text-disabled",
                        )
                        html.Span(
                            "입력 파라미터: {{ nt_param_names.join(', ') }} "
                            "({{ nt_param_names.length }}개)",
                            v_show=("nt_case_mode",),
                            classes="text-caption text-disabled",
                        )
                        self._tip(
                            "{{ nt_case_mode "
                            "? '케이스 세트의 파라미터 표에서 왔습니다.' "
                            ": '단일 케이스 시계열의 유일한 파라미터라 자동으로 "
                            "정해집니다.' }} 위에서 다른 필드를 입력으로 주면 "
                            "(좌표+입력장+시간)→출력 의 field-to-field 연산자가 "
                            "됩니다 (예: U → p). 출력으로 고른 필드는 자동 "
                            "제외됩니다."
                        )
                    v3.VDivider(classes="mb-3")
                    html.Div("모델 방식", classes="text-caption text-disabled mb-1")
                    method_cards = [
                        (
                            "rom",
                            "축소+보간 (ROM)",
                            "POD 로 압축 후 계수 보간 · 적은 스냅샷 · 모든 메쉬 · 표준",
                            "mdi-chart-timeline-variant",
                        ),
                        (
                            "physics",
                            "직접 회귀 (Physics AI)",
                            "좌표+시간→물리량 신경망 · 메쉬 프리 · NVIDIA PhysicsNeMo",
                            "mdi-atom-variant",
                        ),
                        (
                            "operator",
                            "신경 연산자 (FNO)",
                            "함수→함수 · 다수 샘플 · 균일 격자 · ms 추론",
                            "mdi-waveform",
                        ),
                        (
                            "dynamics",
                            "동역학 예보 (DMD)",
                            "시간 전이 규칙 학습 · 학습 구간 밖 외삽 가능 · PyDMD",
                            "mdi-chart-bell-curve-cumulative",
                        ),
                    ]
                    for key, name, subtitle, icon in method_cards:
                        # 능력 레지스트리 판정 (v5.0) — 불가면 카드를 흐리게 하고
                        # 이유를 카드 안에 쓴다. 학습 버튼을 눌러야 알던 것을
                        # 로드 시점에 미리 보여주는 것.
                        infeasible = (
                            f"nt_strategy_status['{key}'] && "
                            f"!nt_strategy_status['{key}'].ok"
                        )
                        with v3.VCard(
                            classes="mb-1",
                            click=f"nt_model_method = '{key}'",
                            variant=(
                                f"nt_model_method === '{key}' ? 'tonal' : 'outlined'",
                            ),
                            color=(
                                f"nt_model_method === '{key}' ? 'primary' : undefined",
                            ),
                            style=(f"{infeasible} ? 'opacity:0.55' : ''",),
                        ):
                            with v3.VCardText(classes="py-2 d-flex align-center"):
                                v3.VIcon(icon, classes="mr-3", size="small")
                                with html.Div():
                                    html.Div(name, classes="text-body-2")
                                    html.Div(
                                        subtitle,
                                        classes="text-caption text-disabled",
                                    )
                                    html.Div(
                                        f"{{{{ nt_strategy_status['{key}'] ? "
                                        f"nt_strategy_status['{key}'].reason : '' }}}}",
                                        v_show=(infeasible,),
                                        classes="text-caption text-warning",
                                    )
                    # 데이터 기반 자동 추천 (service.recommend_method)
                    html.Div(
                        "{{ nt_method_hint }}",
                        v_show=("nt_method_hint",),
                        classes="text-caption text-info mt-1 mb-2",
                    )

                    # Ⓐ ROM: reducer × 계수 회귀
                    with html.Div(v_show=("nt_model_method === 'rom'",)):
                        v3.VSelect(
                            v_model=("nt_reducer",),
                            items=("nt_reducer_choices",),
                            label="Reducer (차원 축소)",
                            density="compact",
                            classes="mt-1",
                        )
                        v3.VSelect(
                            v_model=("nt_surrogate",),
                            items=("nt_surrogate_choices",),
                            label="계수 회귀 (Surrogate)",
                            density="compact",
                            classes="mt-2",
                        )

                    # Ⓑ Physics AI: 직접 회귀 파라미터
                    with html.Div(v_show=("nt_model_method === 'physics'",)):
                        self._tip_row(
                            "학습 설정",
                            "POD reducer 없이 좌표+시간을 필드로 직접 매핑합니다 "
                            "(torch 만으로 학습 — physicsnemo 패키지는 ④Export "
                            "모듈 저장에만 필요).",
                        )
                        v3.VTextField(
                            v_model=("nt_physics_epochs",),
                            label="Epochs",
                            type="number",
                            density="compact",
                            classes="mt-1",
                        )
                        v3.VTextField(
                            v_model=("nt_physics_hidden",),
                            label="Hidden width",
                            type="number",
                            density="compact",
                            classes="mt-2",
                        )
                        v3.VTextField(
                            v_model=("nt_physics_max_samples",),
                            label="Max train samples",
                            type="number",
                            density="compact",
                            classes="mt-2",
                        )

                    # Ⓒ 신경 연산자: ⑥연산자 랩으로 안내 (로드 데이터 직학습은 P4)
                    with html.Div(v_show=("nt_model_method === 'operator'",)):
                        # 로드한 데이터로는 아직 학습 불가 — 제약이라 ⚠.
                        self._tip_row(
                            "내 데이터 직접 학습은 아직 미지원",
                            "신경 연산자는 다수 샘플(수백+)·균일 격자 데이터에 "
                            "적합합니다 (균일 격자: FNO — 탑재됨 · 기하 인지: "
                            "GNN/GINO — 예정). 현재는 ⑥연산자 랩의 표준 벤치마크 "
                            "문제로만 학습할 수 있습니다.",
                            warn=True,
                        )
                        v3.VBtn(
                            "⑥ 연산자 랩 열기",
                            click="nt_open_panels = [5]",
                            variant="tonal",
                            block=True,
                            classes="mt-1",
                            prepend_icon="mdi-open-in-app",
                        )

                    # Ⓓ 동역학 예보 (PyDMD) — 학습 구간 밖 외삽이 가능한 유일 계열.
                    with html.Div(v_show=("nt_model_method === 'dynamics'",)):
                        # DMD 는 부적합해도 조용히 학습에 "성공"한다 — 못 보면
                        # 결과가 조용히 틀어지므로 ⚠ 로 신호를 남긴다.
                        self._tip_row(
                            "데이터가 맞아야만 쓸 수 있습니다",
                            "상태의 시간 전이 규칙을 학습해 학습 구간 밖까지 "
                            "예보합니다. 유동이 '공간모드 × 고유 주파수'의 저랭크 "
                            "선형 동역학으로 근사될 때만 맞습니다 — 강한 이류나 "
                            "불연속(필라멘트 데모 등)에는 부적합하니 학습 후 "
                            "적합도(재구성 오차)를 반드시 확인하세요. 모드 수는 "
                            "PyDMD 가 자동 결정합니다 (실수 진동은 켤레쌍 때문에 "
                            "물리 모드당 랭크 2가 필요해 수동 지정 시 과소적합되기 "
                            "쉽습니다).",
                            warn=True,
                        )
                        v3.VSelect(
                            v_model=("nt_dmd_method",),
                            items=("nt_dmd_choices",),
                            label="DMD 변형",
                            density="compact",
                            classes="mt-1",
                        )
                        with v3.VCard(
                            variant="tonal", classes="mt-2", v_show=("nt_dmd_ready",)
                        ):
                            with v3.VCardText(classes="text-caption"):
                                html.Div("{{ nt_dmd_summary }}")
                                # 적합도 신호등 — DMD 는 안 맞아도 조용히 틀린다.
                                html.Div(
                                    "적합도: 재구성 오차 "
                                    "{{ (nt_dmd_fit_error * 100).toFixed(1) }}%",
                                    classes=(
                                        "nt_dmd_fit_error < 0.1 ? 'text-success mt-1'"
                                        " : (nt_dmd_fit_error < 0.3 ?"
                                        " 'text-warning mt-1' : 'text-error mt-1')",
                                    ),
                                )
                                html.Div(
                                    "재구성 오차가 크면 이 데이터에 DMD 가 맞지 "
                                    "않는 것입니다 — 예보를 믿지 마세요.",
                                    v_show=("nt_dmd_fit_error >= 0.3",),
                                    classes="text-error",
                                )

                    # 라벨은 입력 종류에 따라 바뀐다 (t vs 운전조건). VBtn 의
                    # 첫 위치 인자는 텍스트 child 라 바인딩이 안 되므로 mustache 로.
                    with html.Div(
                        classes="d-flex align-center mt-2",
                        v_show=("nt_model_method !== 'operator'",),
                    ):
                        with v3.VBtn(
                            click=self.ctrl.nt_model_train,
                            color="primary",
                            block=True,
                            disabled=(
                                "(!nt_has_timesteps && !nt_case_mode) || nt_busy",
                            ),
                            prepend_icon="mdi-cog-sync-outline",
                            classes="flex-grow-1",
                        ):
                            html.Span(
                                "{{ nt_case_mode ? '모델 학습 (운전조건→필드)' "
                                ": '모델 학습 (시간→필드)' }}"
                            )
                        # 버튼이 비활성일 때가 바로 이 설명이 필요한 순간이므로,
                        # 아이콘은 버튼 밖에 둔다 (disabled 는 hover 가 죽는다).
                        self._tip(
                            "{{ nt_case_mode "
                            "? '케이스 세트의 운전조건으로 학습합니다.' "
                            ": '2개 이상 타임스텝이 필요합니다.' }} "
                            "모드 수는 ⑤부가 분석의 슬라이더와 공유합니다."
                        )
                    # "케이스 N개로 학습" 은 상태(무엇으로 학습하는지)라 남긴다.
                    html.Div(
                        "케이스 {{ nt_case_count }}개로 학습합니다.",
                        classes="text-caption text-disabled mt-1",
                        v_show=("nt_model_method !== 'operator' && nt_case_mode",),
                    )
                    with v3.VCard(variant="tonal", classes="mt-3", v_show=("nt_model_ready",)):
                        with v3.VCardText(classes="text-caption"):
                            html.Div("학습 완료 — {{ nt_model_summary }}")

                    # 자동 비교 리더보드 (구 ⑦Compare 흡수) — 내 데이터에서
                    # ROM 조합 + Physics AI 를 같은 지표로 순위 매기는 모델 선정.
                    v3.VDivider(classes="my-4")
                    html.Div(
                        "자동 비교 (리더보드)",
                        classes="text-caption text-disabled mb-1",
                    )
                    with html.Div(classes="d-flex align-center"):
                        v3.VBtn(
                            "전체 방식 비교",
                            click=self.ctrl.nt_run_compare,
                            variant="tonal",
                            color="primary",
                            block=True,
                            disabled=("!nt_has_timesteps || nt_case_mode || nt_busy",),
                            prepend_icon="mdi-table-search",
                            classes="flex-grow-1",
                        )
                        # 케이스 세트에서는 버튼이 비활성이라 "왜 안 되는지"를
                        # 알려주는 게 이 아이콘의 존재 이유다 — 그래서 ⚠.
                        self._tip(
                            "ROM 조합(POD×RBF/Kriging) + Physics AI 를 RMSE·R²·"
                            "지연시간으로 순위 비교합니다 (모드 수는 ⑤부가 분석 "
                            "공유).",
                            v_show_expr="!nt_case_mode",
                        )
                        self._tip(
                            "케이스 세트(파라미터 스윕)의 자동 비교는 아직 지원하지 "
                            "않습니다 — ②Model 에서 방식을 바꿔가며 직접 학습해 "
                            "비교하세요.",
                            warn=True,
                            v_show_expr="nt_case_mode",
                        )
                    with v3.VCard(
                        variant="tonal", classes="mt-2", v_show=("nt_compare_summary",)
                    ):
                        with v3.VCardText(classes="text-caption"):
                            html.Div("{{ nt_compare_summary }}")
                            v3.VBtn(
                                "결과 표 다시 보기",
                                click="nt_compare_dialog = true",
                                variant="text",
                                size="small",
                                classes="mt-1",
                            )

            # 3) Twin
            with v3.VExpansionPanel(title="③ Twin (입력→필드 예측)"):
                with v3.VExpansionPanelText():
                    html.Div(
                        "먼저 ②Model 에서 학습하세요.",
                        classes="text-caption text-disabled",
                        v_show=("!nt_twin_ready",),
                    )
                    with v3.VCard(variant="flat", v_show=("nt_twin_ready",)):
                        with v3.VCardText():
                            html.Div("{{ nt_twin_summary }}", classes="text-caption mb-2")
                            # 문제 유형 A — 시간 슬라이더 1개.
                            with html.Div(v_show=("!nt_case_mode",)):
                                html.Div(
                                    "예측 파라미터 t = {{ nt_twin_param }}",
                                    classes="text-caption mb-1",
                                )
                                v3.VSlider(
                                    v_model=("nt_twin_param",),
                                    min=("nt_twin_min",),
                                    max=("nt_twin_max",),
                                    step=("nt_twin_step",),
                                    hide_details=True,
                                    density="compact",
                                    # 외삽 구간에 들어가면 색으로 경고 (DMD 전용).
                                    color=(
                                        "nt_dmd_ready && nt_twin_param > "
                                        "nt_twin_train_max ? 'warning' : 'primary'",
                                    ),
                                )
                                html.Div(
                                    "⚠ 학습 구간 밖 — 외삽 예보입니다 "
                                    "(학습 t ≤ {{ nt_twin_train_max }})",
                                    v_show=(
                                        "nt_dmd_ready && nt_twin_param > "
                                        "nt_twin_train_max",
                                    ),
                                    classes="text-caption text-warning",
                                )
                            # 문제 유형 B — 운전조건 슬라이더 k 개.
                            # 배열 원소를 v-model 로 바꾸면 trame 이 dirty 를
                            # 감지 못하므로 flushState 로 명시 push 한다.
                            with html.Div(v_show=("nt_case_mode",)):
                                with html.Template(
                                    v_for="(pname, i) in nt_param_names",
                                    key="pname",
                                ):
                                    html.Div(
                                        "{{ pname }} = {{ nt_twin_params[i] }}",
                                        classes="text-caption mb-1",
                                    )
                                    v3.VSlider(
                                        v_model=("nt_twin_params[i]",),
                                        min=("nt_twin_mins[i]",),
                                        max=("nt_twin_maxs[i]",),
                                        step=("nt_twin_steps[i]",),
                                        hide_details=True,
                                        density="compact",
                                        classes="mb-1",
                                        raw_attrs=[
                                            '@update:model-value='
                                            '"flushState(\'nt_twin_params\')"'
                                        ],
                                    )
                            v3.VBtn(
                                "예측 실행",
                                click=self.ctrl.nt_predict,
                                color="secondary",
                                block=True,
                                classes="mt-2",
                                disabled=("nt_busy",),
                                prepend_icon="mdi-play",
                            )
                            # 출력 격자 자유화(M3) — Physics AI(신경장)만 가능.
                            v3.VDivider(classes="my-3")
                            self._tip_row(
                                "출력 격자",
                                "신경장(Physics AI)은 좌표를 입력으로 받아 학습 "
                                "격자에 묶이지 않습니다 — 더 촘촘한 격자나 다른 "
                                "메쉬 파일에 그대로 예측할 수 있습니다. ROM 은 "
                                "POD 모드가 학습 메쉬에 묶여 불가합니다.",
                            )
                            html.Div(
                                "현재: {{ nt_predict_mesh_name || '학습 격자' }}",
                                classes="text-caption mb-1",
                            )
                            v3.VBtn(
                                "다른 격자에 예측",
                                click=self.ctrl.nt_fb_open_predict_mesh,
                                variant="tonal",
                                block=True,
                                disabled=("!nt_physics_ready || nt_busy",),
                                prepend_icon="mdi-grid",
                            )
                            v3.VBtn(
                                "학습 격자로 복귀",
                                click=self.ctrl.nt_restore_training_mesh,
                                variant="text",
                                block=True,
                                classes="mt-1",
                                v_show=("nt_predict_mesh_name",),
                                prepend_icon="mdi-undo",
                            )

            # 4) Export
            with v3.VExpansionPanel(title="④ Export (저장)"):
                with v3.VExpansionPanelText():
                    v3.VTextField(
                        v_model=("nt_export_dir",),
                        label="저장 폴더",
                        density="compact",
                        clearable=True,
                    )
                    with v3.VRow(classes="mt-1", dense=True):
                        with v3.VCol(cols=6):
                            v3.VBtn(
                                "스크린샷",
                                click=self.ctrl.nt_export_screenshot,
                                variant="tonal",
                                block=True,
                                size="small",
                                disabled=("!nt_has_dataset",),
                            )
                        with v3.VCol(cols=6):
                            v3.VBtn(
                                "필드 CSV",
                                click=self.ctrl.nt_export_csv,
                                variant="tonal",
                                block=True,
                                size="small",
                                disabled=("!nt_has_dataset",),
                            )
                        with v3.VCol(cols=6):
                            v3.VBtn(
                                "메쉬 VTK",
                                click=self.ctrl.nt_export_vtk,
                                variant="tonal",
                                block=True,
                                size="small",
                                disabled=("!nt_has_dataset",),
                            )
                        with v3.VCol(cols=6):
                            v3.VBtn(
                                ".ntwin",
                                click=self.ctrl.nt_export_project,
                                variant="tonal",
                                block=True,
                                size="small",
                                disabled=("!nt_has_dataset",),
                            )
                        with v3.VCol(cols=6):
                            v3.VBtn(
                                "엔진 .pkl",
                                click=self.ctrl.nt_export_engine,
                                variant="tonal",
                                block=True,
                                size="small",
                                disabled=("!nt_model_ready",),
                            )
                        with v3.VCol(cols=6):
                            v3.VBtn(
                                "보고서",
                                click=self.ctrl.nt_export_report,
                                variant="tonal",
                                block=True,
                                size="small",
                                disabled=("!nt_has_dataset",),
                            )
                        with v3.VCol(cols=6):
                            v3.VBtn(
                                "PhysicsNeMo Module",
                                click=self.ctrl.nt_export_physicsnemo,
                                variant="tonal",
                                block=True,
                                size="small",
                                disabled=("!nt_physics_ready",),
                            )
                    with v3.VCard(variant="tonal", classes="mt-3", v_show=("nt_export_last",)):
                        with v3.VCardText(classes="text-caption"):
                            html.Div("최근 저장: {{ nt_export_last }}")

            html.Div(
                "보조 도구 (선택 — 트윈 학습에는 필요 없음)",
                classes="text-overline text-disabled px-1 mt-4 mb-1",
            )
            # 5) 부가 분석 (선택) — 구 ②Analyze + ③Reduce 통합.
            # 둘 다 ②Model 학습과 무관한 진단/탐색 도구다 (build_twin() 은
            # nt_analysis_done/nt_pod_done 을 참조하지 않고 POD 를 내부에서
            # 새로 만든다) — 근거·판단: .omc/plans/model-taxonomy-plan.md §9.
            with v3.VExpansionPanel(title="⑤ 부가 분석 (선택)"):
                with v3.VExpansionPanelText():
                    with html.Div(classes="d-flex align-center mb-1"):
                        html.Span("와류 식별 (Vortex ID)", classes="text-subtitle-2")
                        self._tip(
                            "②Model 트윈 학습에는 필요 없는 진단 도구입니다 — "
                            "유동장을 더 깊이 들여다보고 싶을 때만 쓰세요."
                        )
                    v3.VSelect(
                        v_model=("nt_method",),
                        items=("nt_method_choices",),
                        label="기법",
                        density="compact",
                    )
                    with html.Div(classes="d-flex align-center mt-2"):
                        v3.VBtn(
                            "분석 실행",
                            click=self.ctrl.nt_run_analysis,
                            color="primary",
                            block=True,
                            disabled=("!nt_has_dataset || nt_busy",),
                            prepend_icon="mdi-tornado",
                            classes="flex-grow-1",
                        )
                        self._tip(
                            "속도장 'U' 가 필요합니다 (Q-criterion / λ₂).",
                            warn=True,
                        )
                    v3.VDivider(classes="my-3")
                    html.Div("FFT / PSD (시계열 주파수 분석)", classes="text-caption mb-1")
                    v3.VSelect(
                        v_model=("nt_fft_field",),
                        items=("nt_fields",),
                        label="FFT field (빈 값=기본)",
                        density="compact",
                        hide_details=True,
                        clearable=True,
                        disabled=("!nt_has_timesteps",),
                    )
                    v3.VTextField(
                        v_model=("nt_fft_dt",),
                        label="dt [s] (0=자동)",
                        type="number",
                        density="compact",
                        hide_details=True,
                        classes="mt-2",
                    )
                    v3.VSwitch(
                        v_model=("nt_fft_probe",),
                        label="프로브점 (off=공간 평균)",
                        density="compact",
                        hide_details=True,
                        color="primary",
                        classes="mt-1",
                    )
                    with v3.VCard(variant="flat", classes="mt-1", v_show=("nt_fft_probe",)):
                        with v3.VCardText():
                            html.Div(
                                "프로브 인덱스: {{ nt_fft_point }} / {{ nt_info_points - 1 }}",
                                classes="text-caption mb-1",
                            )
                            v3.VSlider(
                                v_model=("nt_fft_point",),
                                min=0,
                                max=("nt_info_points - 1",),
                                step=1,
                                hide_details=True,
                                density="compact",
                            )
                    v3.VBtn(
                        "FFT/PSD 분석",
                        click=self.ctrl.nt_run_fft,
                        color="secondary",
                        block=True,
                        classes="mt-2",
                        disabled=("!nt_has_timesteps || nt_busy",),
                        prepend_icon="mdi-sine-wave",
                    )
                    html.Div(
                        "{{ nt_fft_summary }}",
                        classes="text-caption text-disabled mt-1",
                        v_show=("nt_fft_summary",),
                    )

                    v3.VDivider(classes="my-3")
                    html.Div("POD 진단 (모드 에너지)", classes="text-subtitle-2 mb-1")
                    v3.VSlider(
                        v_model=("nt_n_modes",),
                        label="모드 수",
                        min=1,
                        max=20,
                        step=1,
                        thumb_label=True,
                        density="compact",
                    )
                    v3.VBtn(
                        "POD 실행",
                        click=self.ctrl.nt_run_pod,
                        color="primary",
                        block=True,
                        classes="mt-2",
                        disabled=("!nt_has_dataset || nt_busy",),
                        prepend_icon="mdi-chart-line",
                    )
                    with v3.VCard(variant="tonal", classes="mt-3", v_show=("nt_pod_done",)):
                        with v3.VCardText(classes="text-caption"):
                            html.Div("{{ nt_pod_summary }}")
                            html.Div(
                                "누적 에너지(%): {{ nt_pod_energy }}",
                                classes="mt-1",
                            )
                    v3.VBtn(
                        "에너지 스펙트럼 차트",
                        click=self.ctrl.nt_show_energy,
                        variant="tonal",
                        block=True,
                        classes="mt-2",
                        v_show=("nt_pod_done",),
                        prepend_icon="mdi-poll",
                    )
                    # POD 모드 3D 시각화
                    with v3.VCard(variant="flat", classes="mt-2", v_show=("nt_pod_done",)):
                        with v3.VCardText():
                            html.Div(
                                "POD 모드 #{{ nt_pod_mode }} 형상 보기",
                                classes="text-caption mb-1",
                            )
                            v3.VSlider(
                                v_model=("nt_pod_mode",),
                                min=0,
                                max=("nt_pod_max_mode",),
                                step=1,
                                hide_details=True,
                                density="compact",
                            )
                            v3.VBtn(
                                "모드 3D 표시",
                                click=self.ctrl.nt_view_pod_mode,
                                variant="tonal",
                                block=True,
                                classes="mt-2",
                                disabled=("nt_busy",),
                                prepend_icon="mdi-waveform",
                            )

            # 6) 연산자 랩 (구 ⑧ AI Bench) — 표준 벤치마크 문제 실험실.
            # 구 ⑦Compare(내 데이터 모델 선정)는 ②Model 의 "자동 비교" 섹션으로
            # 흡수됨 — 근거: .omc/plans/model-taxonomy-plan.md §8.
            with v3.VExpansionPanel(title="⑥ 연산자 랩 (Benchmark Lab)"):
                with v3.VExpansionPanelText():
                    # A) 데이터 소스 — 로드한 데이터와 무관하다는 점이 이 패널의
                    # 가장 큰 오해 지점이라 ⚠ 로 남긴다.
                    with html.Div(classes="d-flex align-center"):
                        v3.VSelect(
                            v_model=("nt_bench_kind",),
                            items=("nt_bench_choices",),
                            label="내장 벤치마크",
                            density="compact",
                            hide_details=True,
                            classes="flex-grow-1",
                        )
                        self._tip(
                            "로드한 데이터와 무관한 표준 벤치마크 문제(Burgers/"
                            "열전도/공동 유동)로 신경 연산자(FNO)를 실험하는 "
                            "공간입니다. 내 데이터 모델 비교는 ②Model 의 "
                            "'자동 비교'를 쓰세요.",
                            warn=True,
                        )
                    with v3.VRow(classes="mt-1", dense=True):
                        with v3.VCol(cols=6):
                            v3.VTextField(
                                v_model=("nt_bench_nsamples",),
                                label="샘플 수",
                                type="number",
                                density="compact",
                                hide_details=True,
                            )
                        with v3.VCol(cols=6):
                            v3.VTextField(
                                v_model=("nt_bench_nx",),
                                label="해상도 N",
                                type="number",
                                density="compact",
                                hide_details=True,
                            )
                    v3.VBtn(
                        "데이터셋 생성 (내장 솔버)",
                        click=self.ctrl.nt_bench_generate,
                        color="primary",
                        block=True,
                        classes="mt-2",
                        disabled=("nt_busy",),
                        prepend_icon="mdi-database-plus-outline",
                    )
                    v3.VTextField(
                        v_model=("nt_bench_path",),
                        label="PDEBench HDF5 경로 (.h5/.hdf5)",
                        density="compact",
                        classes="mt-3",
                        clearable=True,
                        hide_details=True,
                    )
                    v3.VBtn(
                        "PDEBench 파일 로드",
                        click=self.ctrl.nt_bench_load,
                        variant="tonal",
                        block=True,
                        classes="mt-2",
                        disabled=("nt_busy",),
                        prepend_icon="mdi-database-import-outline",
                    )
                    with v3.VCard(variant="tonal", classes="mt-2", v_show=("nt_bench_ready",)):
                        with v3.VCardText(classes="text-caption"):
                            html.Div("{{ nt_bench_summary }}")
                    # B) 연산자 학습
                    v3.VDivider(classes="my-3")
                    # FNO 구현 선택 — 같은 계약이라 동일 조건으로 직접 비교된다.
                    with html.Div(classes="d-flex align-center mb-2"):
                        v3.VSelect(
                            v_model=("nt_bench_backend",),
                            items=("nt_bench_backend_choices",),
                            label="FNO 구현",
                            density="compact",
                            hide_details=True,
                            classes="flex-grow-1",
                        )
                        self._tip(
                            "neuraloperator 는 FNO 논문 저자들이 유지하는 레퍼런스 "
                            "구현입니다. 같은 벤치·하이퍼파라미터로 자체 구현과 "
                            "바꿔가며 비교할 수 있습니다."
                        )
                    with v3.VRow(dense=True):
                        with v3.VCol(cols=4):
                            v3.VTextField(
                                v_model=("nt_bench_epochs",),
                                label="Epochs",
                                type="number",
                                density="compact",
                                hide_details=True,
                            )
                        with v3.VCol(cols=4):
                            v3.VTextField(
                                v_model=("nt_bench_modes",),
                                label="Modes",
                                type="number",
                                density="compact",
                                hide_details=True,
                            )
                        with v3.VCol(cols=4):
                            v3.VTextField(
                                v_model=("nt_bench_width",),
                                label="Width",
                                type="number",
                                density="compact",
                                hide_details=True,
                            )
                    v3.VBtn(
                        "FNO 연산자 학습",
                        click=self.ctrl.nt_bench_train,
                        color="primary",
                        block=True,
                        classes="mt-2",
                        disabled=("!nt_bench_ready || nt_busy",),
                        prepend_icon="mdi-brain",
                    )
                    # 라이브 학습 진행 — epoch별 진행바 + 손실 스파크라인
                    with v3.VCard(
                        variant="tonal", classes="mt-2", v_show=("nt_bench_training",)
                    ):
                        with v3.VCardText():
                            html.Div(
                                "epoch {{ nt_bench_epoch }} / {{ nt_bench_epochs_total }} · "
                                "loss {{ nt_bench_loss.toExponential(3) }}",
                                classes="text-caption nt-mono mb-1",
                            )
                            v3.VProgressLinear(
                                model_value=("nt_progress < 0 ? 0 : nt_progress",),
                                color="primary",
                                height=6,
                                rounded=True,
                                striped=True,
                            )
                            # 라이브 손실 스파크라인 (유니코드 블록 — 클라이언트 의존 없음)
                            html.Div(
                                "{{ nt_bench_loss_series.length > 1 ? "
                                "nt_bench_loss_series.slice(-48).map(v => "
                                "'\\u2581\\u2582\\u2583\\u2584\\u2585\\u2586\\u2587\\u2588'"
                                "[Math.max(0,Math.min(7,Math.round(7*(v-Math.min(...nt_bench_loss_series))/"
                                "((Math.max(...nt_bench_loss_series)-Math.min(...nt_bench_loss_series))||1))))]"
                                ").join('') : '' }}",
                                classes="nt-mono text-info mt-2",
                                style="font-size:1.05rem; line-height:1; letter-spacing:-1px;",
                                v_show=("nt_bench_loss_series.length > 1",),
                            )
                    with v3.VCard(variant="tonal", classes="mt-2", v_show=("nt_bench_trained",)):
                        with v3.VCardText(classes="text-caption"):
                            html.Div("{{ nt_bench_train_summary }}")
                    # C) 샘플 예측 비교
                    with v3.VCard(variant="flat", classes="mt-2", v_show=("nt_bench_trained",)):
                        with v3.VCardText():
                            html.Div(
                                "샘플 #{{ nt_bench_sample }} / {{ nt_bench_max_sample }}",
                                classes="text-caption mb-1",
                            )
                            v3.VSlider(
                                v_model=("nt_bench_sample",),
                                min=0,
                                max=("nt_bench_max_sample",),
                                step=1,
                                hide_details=True,
                                density="compact",
                            )
                            v3.VBtn(
                                "참값 vs 예측 비교",
                                click=self.ctrl.nt_bench_eval,
                                color="secondary",
                                block=True,
                                classes="mt-2",
                                disabled=("nt_busy",),
                                prepend_icon="mdi-chart-bell-curve",
                            )


def create_web_app(*, build_ui: bool = True, server: Any = None) -> NavierTwinWebApp:
    """NavierTwin 웹 앱을 생성한다.

    Args:
        build_ui: True 면 trame UI(레이아웃 + PyVista 뷰어)까지 구성한다.
            테스트 등 GL 컨텍스트가 없는 환경에서는 False 로 두면 state/
            controller 만 초기화한다.
        server: 사용할 기존 trame server (선택).

    Returns:
        구성된 :class:`NavierTwinWebApp`.
    """
    app = NavierTwinWebApp(server=server)
    if build_ui:
        app.build_ui()
    return app


def run_web(host: str = "127.0.0.1", port: int = 8080, *, open_browser: bool = True) -> int:
    """웹 서버를 시작한다 (로컬 단일 사용자).

    Args:
        host: 바인드 호스트.
        port: 포트.
        open_browser: 시작 시 기본 브라우저 열기.

    Returns:
        프로세스 종료 코드.
    """
    app = create_web_app(build_ui=True)
    app.server.start(host=host, port=port, open_browser=open_browser)
    return 0


__all__ = ["NavierTwinWebApp", "create_web_app", "run_web"]
