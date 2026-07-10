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

        # Import
        st.nt_path = ""
        # 파일 브라우저 모달
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

        # Model
        st.nt_reducer = "pod"
        st.nt_reducer_choices = [
            {"title": "POD (snapshot)", "value": "pod"},
            {"title": "Randomized POD", "value": "randomized_pod"},
        ]
        st.nt_surrogate = "rbf"
        st.nt_surrogate_choices = [
            {"title": "RBF", "value": "rbf"},
            {"title": "Kriging (GP)", "value": "kriging"},
        ]
        st.nt_model_ready = False
        st.nt_model_summary = ""

        # Twin
        st.nt_twin_ready = False
        st.nt_twin_min = 0.0
        st.nt_twin_max = 1.0
        st.nt_twin_param = 0.0
        st.nt_twin_step = 0.01
        st.nt_twin_summary = ""

        # Compare (reducer×surrogate 벤치마크)
        st.nt_compare_dialog = False
        st.nt_compare_rows = []
        st.nt_compare_summary = ""
        st.nt_compare_headers = [
            {"title": "조합", "key": "combo"},
            {"title": "모드", "key": "n_modes", "align": "end"},
            {"title": "RMSE", "key": "rmse", "align": "end"},
            {"title": "R²", "key": "r2", "align": "end"},
            {"title": "rel.L2", "key": "rel_l2", "align": "end"},
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
        ctrl.nt_fb_open = self.fb_open
        ctrl.nt_fb_navigate = self.fb_navigate
        ctrl.nt_fb_pick = self.fb_pick
        ctrl.nt_fb_load_cwd = self.fb_load_cwd
        ctrl.nt_run_analysis = A(self.run_analysis, "와류 식별 계산 중…", render_after=True)
        ctrl.nt_run_fft = A(self.run_fft, "FFT/PSD 계산 중…")
        ctrl.nt_run_pod = A(self.run_pod, "POD 계산 중…")
        ctrl.nt_view_pod_mode = A(self.view_pod_mode, "POD 모드 렌더 중…", render_after=True)
        ctrl.nt_model_train = A(self.build_twin, "트윈 학습 중…")
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
        ctrl.nt_export_report = self.export_report
        self.state.change("nt_field", "nt_cmap", "nt_show_edges", "nt_timestep")(
            self._on_view_state_change
        )

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
        """불연속·비주기 소용돌이 필라멘트 데모 데이터셋을 로드한다."""
        try:
            dataset = service.make_demo_dataset(kind="filament")
            self._set_dataset(
                dataset,
                status="데모 데이터셋 로드 완료 (소용돌이 필라멘트 — 불연속·비주기).",
            )
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
        """파일 브라우저 모달을 연다 (마지막 위치 또는 홈)."""
        self._fb_refresh((self.state.nt_fb_cwd or "").strip() or None)
        self.state.nt_fb_dialog = True

    def fb_navigate(self, path: str) -> None:
        """모달 내에서 디렉토리를 이동한다."""
        self._fb_refresh(path)

    def _fb_dispatch_load(self, path: str) -> None:
        """모달을 닫고 경로 로드를 비동기로 트리거한다 (없으면 동기 폴백)."""
        import asyncio

        self.state.nt_path = path
        self.state.nt_fb_dialog = False
        is_proj = path.lower().endswith(".ntwin")
        msg = ".ntwin 프로젝트 로드 중…" if is_proj else "CFD 데이터 로드 중…"
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            loop.create_task(self._run_async(self.load_path, msg, render_after=True))
        else:  # 테스트 등 루프 밖 컨텍스트
            self.load_path()

    def fb_pick(self, path: str) -> None:
        """브라우저에서 파일을 선택 → 확장자에 따라 로드한다."""
        self._fb_dispatch_load(path)

    def fb_load_cwd(self) -> None:
        """현재 폴더 자체를 로드한다 (OpenFOAM case 디렉토리 등)."""
        cwd = (self.state.nt_fb_cwd or "").strip()
        if cwd:
            self._fb_dispatch_load(cwd)

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
            self.state.nt_model_ready = True
            self.state.nt_twin_ready = True
            self.state.nt_twin_min = pmin
            self.state.nt_twin_max = pmax
            self.state.nt_twin_param = 0.5 * (pmin + pmax)
            self.state.nt_twin_step = max((pmax - pmin) / 100.0, 1e-6)
            self.state.nt_model_summary = summary
            self.state.nt_twin_summary = summary
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
            self._fail("POD 없음", RuntimeError("먼저 ③Reduce 에서 POD 를 실행하세요."))
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
            self._fail("POD 없음", RuntimeError("먼저 ③Reduce 에서 POD 를 실행하세요."))
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
        """④Model — 선택한 reducer/surrogate 로 (시간→필드) 트윈을 학습한다."""
        if self.dataset is None:
            self._fail("데이터 없음", RuntimeError("먼저 데이터를 로드하세요."))
            return
        field = self._base_field()
        reducer = self.state.nt_reducer or "pod"
        surrogate = self.state.nt_surrogate or "rbf"
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
                self.state.nt_twin_ready = True
                self.state.nt_twin_min = pmin
                self.state.nt_twin_max = pmax
                self.state.nt_twin_param = 0.5 * (pmin + pmax)
                self.state.nt_twin_step = max((pmax - pmin) / 100.0, 1e-6)
                self.state.nt_twin_summary = summary
            self._set_status(f"모델 학습 완료: {reducer}+{surrogate}. ⑤Twin 에서 예측하세요.")
        except Exception as exc:  # noqa: BLE001
            self._fail("모델 학습 실패", exc)

    def run_compare(self) -> None:
        """모든 reducer×surrogate 조합을 학습/평가해 순위표를 표시한다."""
        if self.dataset is None:
            self._fail("데이터 없음", RuntimeError("먼저 데이터를 로드하세요."))
            return
        try:
            summary = self._apply_compare_result(self._compare_compute(None))
            self._set_status(f"모델 비교 완료 — {summary}")
        except Exception as exc:  # noqa: BLE001
            self._fail("모델 비교 실패", exc)

    def _compare_compute(self, progress_cb: Any) -> dict[str, Any]:
        """전체 reducer×surrogate 조합 벤치마크 (동기 워커 — 상태 미변경)."""
        field = self._base_field()
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
        return {
            "combo": row.get("combo", ""),
            "n_modes": int(row.get("n_modes", 0)),
            "rmse": fmt(row.get("rmse")),
            "r2": fmt(row.get("r2"), "{:.4f}"),
            "rel_l2": fmt(row.get("rel_l2")),
            "latency_ms": fmt(row.get("latency_ms"), "{:.3f}"),
            "status": "ok" if status == "ok" else status[:40],
        }

    # ------------------------------------------------------------------
    # AI Bench callbacks (⑧) — 벤치마크 데이터셋 → 연산자 학습 → 빠른 예측
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
            epochs=int(self.state.nt_bench_epochs or 60),
            modes=int(self.state.nt_bench_modes or 12),
            width=int(self.state.nt_bench_width or 32),
            progress_cb=progress_cb,
        )

    def _apply_bench_result(self, result: dict[str, Any]) -> str:
        """학습 결과를 상태/차트에 반영한다 (loop 스레드에서만 호출)."""
        self._bench_result = result
        summary = (
            f"{result['operator'].upper()} · {result['n_train']}train/{result['n_test']}test · "
            f"loss {result['final_loss']:.3g} · test RMSE {result['test_rmse']:.3g} "
            f"(rel.L2 {result['test_rel_l2']:.3g}) · "
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

    def predict(self) -> None:
        """현재 파라미터(시간)에서 필드를 예측하고 3D 뷰어에 표시한다."""
        if self.engine is None or self.dataset is None:
            self._fail("트윈 없음", RuntimeError("먼저 트윈을 학습하세요."))
            return
        try:
            value = float(self.state.nt_twin_param or 0.0)
            prediction = service.predict_twin(self.engine, value)
            field = service.attach_prediction(self.dataset, prediction)
            self._refresh_fields(prefer=field)
            self._render(reset_camera=False)
            self._set_status(f"예측 완료: t={value:.4g} → '{field}' 3D 표시")
        except Exception as exc:  # noqa: BLE001
            self._fail("예측 실패", exc)

    def reset_view(self) -> None:
        """카메라를 기본 뷰로 리셋한다."""
        self._render(reset_camera=True)

    # ------------------------------------------------------------------
    # Export callbacks (⑥)
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
            self._fail("모델 없음", RuntimeError("먼저 ④Model 에서 학습하세요."))
            return
        try:
            path = service.save_engine(self.engine, self._export_path("engine.pkl"))
            self._export_done(path, "TwinEngine")
        except Exception as exc:  # noqa: BLE001
            self._fail("엔진 저장 실패", exc)

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
        self.dataset = dataset
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
            self.state.nt_analysis_done = False
            self.state.nt_pod_done = False
            self.state.nt_pod_summary = ""
            self.state.nt_pod_energy = []
            self.state.nt_pod_mode = 0
            self.state.nt_pod_max_mode = 0
            self.state.nt_model_ready = False
            self.state.nt_model_summary = ""
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
        self._render(reset_camera=True)
        self._set_status(status)

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
        """POD/Model 대상 field — 파생 결과가 아닌 원본 물리량을 우선한다."""
        names = render.available_fields(self.dataset) if self.dataset is not None else []
        base = [n for n in names if not self._is_derived_field(n)]
        field = self.state.nt_field or ""
        if field and not self._is_derived_field(field):
            return field
        return render.preferred_field(base) if base else field

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
                        html.Span("파일 · 폴더 열기")
                    with v3.VCardSubtitle(classes="text-caption text-truncate pb-2"):
                        html.Span("{{ nt_fb_cwd }}")
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
                        v3.VBtn(
                            "현재 폴더 로드",
                            click=self.ctrl.nt_fb_load_cwd,
                            variant="tonal",
                            color="primary",
                            prepend_icon="mdi-folder-download-outline",
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
        """드로어 상단 8단계 워크플로우 진행 칩 — 완료 시 초록/체크, 클릭 시 해당 패널 열기."""
        stages = [
            ("①", "Import", "nt_has_dataset", 0),
            ("②", "Analyze", "nt_analysis_done", 1),
            ("③", "Reduce", "nt_pod_done", 2),
            ("④", "Model", "nt_model_ready", 3),
            ("⑤", "Twin", "nt_twin_ready", 4),
            ("⑥", "Export", "!!nt_export_last", 5),
            ("⑦", "Compare", "!!nt_compare_summary", 6),
            ("⑧", "Bench", "nt_bench_trained", 7),
        ]
        with v3.VSheet(color="transparent", classes="d-flex flex-wrap ga-1 px-3 pt-3 pb-1"):
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

    def _build_drawer(self, v3: Any, html: Any) -> None:
        with v3.VExpansionPanels(v_model=("nt_open_panels",), multiple=True):
            # 1) Import
            with v3.VExpansionPanel(title="① Import"):
                with v3.VExpansionPanelText():
                    v3.VBtn(
                        "데모 데이터 로드",
                        click=self.ctrl.nt_load_demo,
                        color="primary",
                        block=True,
                        disabled=("nt_busy",),
                        prepend_icon="mdi-flask-outline",
                    )
                    v3.VBtn(
                        "경로에서 로드",
                        click=self.ctrl.nt_fb_open,
                        variant="tonal",
                        block=True,
                        classes="mt-3",
                        disabled=("nt_busy",),
                        prepend_icon="mdi-folder-search-outline",
                    )
                    html.Div(
                        "탐색기에서 CFD 파일(*.vtk, *.vtu, ...), OpenFOAM 폴더, "
                        "또는 .ntwin 프로젝트를 선택합니다.",
                        classes="text-caption text-disabled mt-1",
                    )
                    html.Div(
                        "같은 폴더의 <name>.engine.pkl 이 있으면 트윈도 함께 복원합니다. "
                        "재로드된 프로젝트는 단일 timestep이라 예측만 가능합니다.",
                        classes="text-caption text-disabled mt-1",
                    )
                    with v3.VCard(variant="tonal", classes="mt-3", v_show=("nt_has_dataset",)):
                        with v3.VCardText(classes="text-caption"):
                            html.Div("Points: {{ nt_info_points }}")
                            html.Div("Cells: {{ nt_info_cells }}")
                            html.Div("Time steps: {{ nt_info_steps }}")
                            html.Div("Fields: {{ nt_info_fields }}")
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

            # 2) Analyze
            with v3.VExpansionPanel(title="② Analyze (와류 식별)"):
                with v3.VExpansionPanelText():
                    v3.VSelect(
                        v_model=("nt_method",),
                        items=("nt_method_choices",),
                        label="기법",
                        density="compact",
                    )
                    v3.VBtn(
                        "분석 실행",
                        click=self.ctrl.nt_run_analysis,
                        color="primary",
                        block=True,
                        classes="mt-2",
                        disabled=("!nt_has_dataset || nt_busy",),
                        prepend_icon="mdi-tornado",
                    )
                    html.Div(
                        "속도장 'U' 가 필요합니다 (Q-criterion / λ₂).",
                        classes="text-caption text-disabled mt-2",
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

            # 3) Reduce
            with v3.VExpansionPanel(title="③ Reduce (POD)"):
                with v3.VExpansionPanelText():
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

            # 4) Model
            with v3.VExpansionPanel(title="④ Model (트윈 학습)"):
                with v3.VExpansionPanelText():
                    v3.VSelect(
                        v_model=("nt_reducer",),
                        items=("nt_reducer_choices",),
                        label="Reducer",
                        density="compact",
                    )
                    v3.VSelect(
                        v_model=("nt_surrogate",),
                        items=("nt_surrogate_choices",),
                        label="Surrogate",
                        density="compact",
                        classes="mt-2",
                    )
                    v3.VBtn(
                        "모델 학습 (시간→필드)",
                        click=self.ctrl.nt_model_train,
                        color="primary",
                        block=True,
                        classes="mt-2",
                        disabled=("!nt_has_timesteps || nt_busy",),
                        prepend_icon="mdi-cog-sync-outline",
                    )
                    html.Div(
                        "2개 이상 타임스텝이 필요합니다 (모드 수는 ③Reduce 슬라이더 공유).",
                        classes="text-caption text-disabled mt-1",
                    )
                    with v3.VCard(variant="tonal", classes="mt-3", v_show=("nt_model_ready",)):
                        with v3.VCardText(classes="text-caption"):
                            html.Div("학습 완료 — {{ nt_model_summary }}")

            # 5) Twin
            with v3.VExpansionPanel(title="⑤ Twin (시간→필드 예측)"):
                with v3.VExpansionPanelText():
                    html.Div(
                        "먼저 ④Model 에서 학습하세요.",
                        classes="text-caption text-disabled",
                        v_show=("!nt_twin_ready",),
                    )
                    with v3.VCard(variant="flat", v_show=("nt_twin_ready",)):
                        with v3.VCardText():
                            html.Div("{{ nt_twin_summary }}", classes="text-caption mb-2")
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

            # 6) Export
            with v3.VExpansionPanel(title="⑥ Export (저장)"):
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
                    with v3.VCard(variant="tonal", classes="mt-3", v_show=("nt_export_last",)):
                        with v3.VCardText(classes="text-caption"):
                            html.Div("최근 저장: {{ nt_export_last }}")

            # 7) Compare
            with v3.VExpansionPanel(title="⑦ Compare (모델 비교)"):
                with v3.VExpansionPanelText():
                    v3.VBtn(
                        "전체 조합 비교",
                        click=self.ctrl.nt_run_compare,
                        color="primary",
                        block=True,
                        disabled=("!nt_has_timesteps || nt_busy",),
                        prepend_icon="mdi-table-search",
                    )
                    html.Div(
                        "POD/Randomized POD × RBF/Kriging 조합을 RMSE·R²·지연시간으로 비교 (모드 수는 ③Reduce 공유).",
                        classes="text-caption text-disabled mt-1",
                    )
                    with v3.VCard(variant="tonal", classes="mt-3", v_show=("nt_compare_summary",)):
                        with v3.VCardText(classes="text-caption"):
                            html.Div("{{ nt_compare_summary }}")
                            v3.VBtn(
                                "결과 표 다시 보기",
                                click="nt_compare_dialog = true",
                                variant="text",
                                size="small",
                                classes="mt-1",
                            )

            # 8) AI Bench — 벤치마크 데이터셋 → 연산자 학습 → 빠른 예측
            with v3.VExpansionPanel(title="⑧ AI Bench (데이터셋→연산자)"):
                with v3.VExpansionPanelText():
                    html.Div(
                        "CFD 벤치마크 데이터로 신경 연산자(FNO)를 학습해 ms 단위 예측을 얻습니다.",
                        classes="text-caption text-disabled mb-2",
                    )
                    # A) 데이터 소스
                    v3.VSelect(
                        v_model=("nt_bench_kind",),
                        items=("nt_bench_choices",),
                        label="내장 벤치마크",
                        density="compact",
                        hide_details=True,
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
