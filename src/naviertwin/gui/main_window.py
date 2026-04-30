"""NavierTwin 메인 윈도우 — 6패널 탭 호스트.

PySide6 기반 메인 애플리케이션 창. Import → Analyze → Reduce →
Model → Twin → Export 순서로 탭이 배치된다.

Examples:
    애플리케이션 시작::

        from PySide6.QtWidgets import QApplication
        from naviertwin.gui.main_window import MainWindow
        import sys

        app = QApplication(sys.argv)
        win = MainWindow()
        win.show()
        sys.exit(app.exec())
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QProcess, QUrl
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from naviertwin import __version__
from naviertwin.gui.panels.analyze_panel import AnalyzePanel
from naviertwin.gui.panels.explainability_panel import ExplainabilityPanel
from naviertwin.gui.panels.export_panel import ExportPanel
from naviertwin.gui.panels.import_panel import ImportPanel
from naviertwin.gui.panels.model_panel import ModelPanel
from naviertwin.gui.panels.reduce_panel import ReducePanel
from naviertwin.gui.panels.twin_panel import TwinPanel
from naviertwin.utils.config import NavierTwinConfig, load_config, save_config
from naviertwin.utils.updater import UpdateCheckResult


def open_file_filter() -> str:
    """File → Open에서 사용할 프로젝트/CFD 통합 파일 필터."""
    from naviertwin.core.cfd_reader import ReaderFactory
    from naviertwin.gui.panels.import_panel import cfd_file_filter

    registered = " ".join(f"*{ext}" for ext in ReaderFactory.registered_extensions())
    return (
        f"All NavierTwin Inputs (*.ntwin {registered});;"
        "NavierTwin Project (*.ntwin);;"
        f"{cfd_file_filter()}"
    )


def format_update_check_message(result: UpdateCheckResult) -> tuple[str, str]:
    """GUI 대화상자용 update-check 결과 메시지를 구성한다."""
    if result.update_available:
        return (
            "업데이트 사용 가능",
            (
                f"현재 버전: {result.current_version}\n"
                f"최신 버전: {result.latest_version}\n"
                f"채널: {result.channel}\n"
                f"다운로드: {result.url}\n"
                f"SHA256: {result.sha256}\n\n"
                "다운로드 열기, URL 복사, 설치파일 검증 버튼으로 업데이트를 안전하게 진행할 수 있습니다."
            ),
        )
    return (
        "최신 상태",
        (
            f"현재 버전: {result.current_version}\n"
            f"확인 채널: {result.channel}\n"
            "사용 가능한 새 업데이트가 없습니다."
        ),
    )


def default_config_path() -> Path:
    """기본 GUI 설정 파일 경로를 반환한다."""
    return Path.home() / ".naviertwin" / "config.json"


def _load_stylesheet(theme: str = "dark") -> str:
    """GUI 테마 QSS를 로드한다."""
    theme_name = theme if theme in {"dark", "light"} else "dark"
    qss_path = Path(__file__).parent / "styles" / f"{theme_name}_theme.qss"
    try:
        return qss_path.read_text(encoding="utf-8")
    except Exception:
        return ""


class MainWindow(QMainWindow):
    """NavierTwin 메인 윈도우.

    6개의 주요 탭을 호스팅하며 탭 간 데이터(CFDDataset, reducer, surrogate,
    TwinEngine)를 시그널/슬롯으로 전달한다.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        confirm_on_close: bool = True,
        config_path: str | Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._confirm_on_close = confirm_on_close
        self._config_path = Path(config_path).expanduser() if config_path else default_config_path()
        self._config = self._load_gui_config()
        self.setWindowTitle("NavierTwin — CFD Digital Twin")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)
        self._latest_dataset: object | None = None
        self._latest_reducer: object | None = None
        self._latest_surrogate: object | None = None
        self._latest_operator: object | None = None
        self._latest_engine: object | None = None
        self._last_acceptance_json: Path | None = None
        self._last_acceptance_summary: Path | None = None
        self._server_process: QProcess | None = None
        self._model_compare_results: dict[str, dict[str, float]] = {}

        self._apply_theme()
        self._setup_panels()
        self._setup_menubar()
        self._setup_statusbar()
        self._connect_signals()

    # ──────────────────────────────────────────────────────────────────
    # 초기화
    # ──────────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        qss = _load_stylesheet(self._config.theme)
        QApplication.instance().setStyleSheet(qss)  # type: ignore[union-attr]

    def _setup_panels(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)

        from naviertwin.utils.i18n import Translator

        self._t = Translator(lang=self._config.language)
        self.setWindowTitle(self._t("app.title", "NavierTwin — CFD Digital Twin"))

        self._import_panel = ImportPanel()
        self._analyze_panel = AnalyzePanel()
        self._reduce_panel = ReducePanel()
        self._model_panel = ModelPanel()
        self._twin_panel = TwinPanel()
        self._export_panel = ExportPanel()
        self._explain_panel: ExplainabilityPanel | None = ExplainabilityPanel()

        # 모델 비교 대시보드 탭
        try:
            from naviertwin.gui.widgets.model_compare_widget import ModelCompareWidget

            self._compare_panel = ModelCompareWidget()
        except Exception:  # noqa: BLE001
            self._compare_panel = None

        # 시뮬레이션 패널 — LBM / Streaming / RL / Burgers
        try:
            from naviertwin.gui.panels.simulation_panel import SimulationPanel

            self._simulation_panel = SimulationPanel()
        except Exception:  # noqa: BLE001
            self._simulation_panel = None

        # Post-Processor Tools 패널 — R591-647 신규 모듈 통합
        try:
            from naviertwin.gui.panels.postproc_panel import PostProcessPanel

            self._postproc_panel = PostProcessPanel()
        except Exception:  # noqa: BLE001
            self._postproc_panel = None

        self._tabs.addTab(self._import_panel,  f"① {self._t('panel.import')}")
        self._tabs.addTab(self._analyze_panel, f"② {self._t('panel.analyze')}")
        self._tabs.addTab(self._reduce_panel,  f"③ {self._t('panel.reduce')}")
        self._tabs.addTab(self._model_panel,   f"④ {self._t('panel.model')}")
        self._tabs.addTab(self._twin_panel,    f"⑤ {self._t('panel.twin')}")
        self._tabs.addTab(self._export_panel,  f"⑥ {self._t('panel.export')}")
        if self._compare_panel is not None:
            self._tabs.addTab(self._compare_panel, "⑦ Compare")
        if self._simulation_panel is not None:
            self._tabs.addTab(self._simulation_panel, "⑧ Simulation")
        if self._explain_panel is not None:
            self._tabs.addTab(self._explain_panel, "⑨ Explain")
        if self._postproc_panel is not None:
            self._tabs.addTab(self._postproc_panel, "⑩ Post-Tools")

        vbox.addWidget(self._tabs)

    def set_language(self, lang: str) -> None:
        """런타임 언어 전환 (탭 제목만 갱신)."""
        self._t.set_language(lang)
        self._config.language = lang  # type: ignore[assignment]
        self.setWindowTitle(self._t("app.title", "NavierTwin — CFD Digital Twin"))
        titles = [
            ("panel.import", "①"),
            ("panel.analyze", "②"),
            ("panel.reduce", "③"),
            ("panel.model", "④"),
            ("panel.twin", "⑤"),
            ("panel.export", "⑥"),
        ]
        for i, (key, num) in enumerate(titles):
            self._tabs.setTabText(i, f"{num} {self._t(key)}")
        self._refresh_view_menu()

    def set_theme(self, theme: str) -> None:
        """런타임 테마 전환."""
        if theme not in {"dark", "light"}:
            raise ValueError(f"지원하지 않는 테마: {theme}")
        self._config.theme = theme  # type: ignore[assignment]
        self._apply_theme()
        self._refresh_view_menu()

    def update_compare_dashboard(
        self, results: dict[str, dict[str, float]]
    ) -> None:
        """외부에서 모델 비교 대시보드 갱신."""
        self._model_compare_results = {
            name: dict(metrics) for name, metrics in results.items()
        }
        if self._compare_panel is not None:
            self._compare_panel.update(self._model_compare_results)

    def _setup_menubar(self) -> None:
        mb: QMenuBar = self.menuBar()

        # 파일 메뉴
        file_menu = mb.addMenu("파일(&F)")

        open_action = QAction("열기(&O)", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)

        self._recent_projects_menu = file_menu.addMenu("최근 프로젝트(&R)")
        self._refresh_recent_projects_menu()

        save_project_action = QAction("프로젝트 저장(&S)", self)
        save_project_action.setShortcut("Ctrl+S")
        save_project_action.triggered.connect(self._save_project)
        file_menu.addAction(save_project_action)

        file_menu.addSeparator()

        quit_action = QAction("종료(&Q)", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # 보기 메뉴
        self._view_menu = mb.addMenu("보기(&V)")
        self._refresh_view_menu()

        # 도구 메뉴
        self._tools_menu = mb.addMenu("도구(&T)")
        benchmark_action = QAction("벤치마크 실행(&B)", self)
        benchmark_action.triggered.connect(self._run_benchmark)
        self._tools_menu.addAction(benchmark_action)

        pipeline_demo_action = QAction("파이프라인 데모 실행(&P)", self)
        pipeline_demo_action.triggered.connect(self._run_pipeline_demo)
        self._tools_menu.addAction(pipeline_demo_action)

        build_twin_action = QAction("CSV 스냅샷으로 트윈 생성(&T)", self)
        build_twin_action.triggered.connect(self._build_twin_from_csv_snapshots)
        self._tools_menu.addAction(build_twin_action)

        predict_twin_action = QAction("저장된 트윈 예측(&R)", self)
        predict_twin_action.triggered.connect(self._predict_twin_from_engine)
        self._tools_menu.addAction(predict_twin_action)

        predict_deployed_twin_action = QAction("배포 트윈 디렉토리 예측(&D)", self)
        predict_deployed_twin_action.triggered.connect(self._predict_twin_from_artifacts_dir)
        self._tools_menu.addAction(predict_deployed_twin_action)

        benchmark_deployed_twin_action = QAction("배포 트윈 지연시간 측정(&L)", self)
        benchmark_deployed_twin_action.triggered.connect(self._benchmark_twin_from_artifacts_dir)
        self._tools_menu.addAction(benchmark_deployed_twin_action)

        validate_twin_action = QAction("저장된 트윈 검증(&V)", self)
        validate_twin_action.triggered.connect(self._validate_twin_from_engine)
        self._tools_menu.addAction(validate_twin_action)

        validate_deployed_twin_action = QAction("배포 트윈 디렉토리 검증(&W)", self)
        validate_deployed_twin_action.triggered.connect(self._validate_twin_from_artifacts_dir)
        self._tools_menu.addAction(validate_deployed_twin_action)

        package_twin_action = QAction("트윈 산출물 패키징(&Z)", self)
        package_twin_action.triggered.connect(self._package_twin_artifacts)
        self._tools_menu.addAction(package_twin_action)

        inspect_twin_package_action = QAction("트윈 패키지 정보 보기(&I)", self)
        inspect_twin_package_action.triggered.connect(self._inspect_twin_package)
        self._tools_menu.addAction(inspect_twin_package_action)

        verify_twin_package_action = QAction("트윈 패키지 검증(&Y)", self)
        verify_twin_package_action.triggered.connect(self._verify_twin_package)
        self._tools_menu.addAction(verify_twin_package_action)

        extract_twin_package_action = QAction("트윈 패키지 검증 후 추출(&U)", self)
        extract_twin_package_action.triggered.connect(self._verify_and_extract_twin_package)
        self._tools_menu.addAction(extract_twin_package_action)

        accept_twin_package_action = QAction("트윈 패키지 원샷 수락 검사(&C)", self)
        accept_twin_package_action.triggered.connect(self._accept_twin_package)
        self._tools_menu.addAction(accept_twin_package_action)

        server_start_action = QAction("API 서버 시작(&S)", self)
        server_start_action.triggered.connect(self._start_api_server)
        self._tools_menu.addAction(server_start_action)

        server_stop_action = QAction("API 서버 중지(&X)", self)
        server_stop_action.triggered.connect(self._stop_api_server)
        self._tools_menu.addAction(server_stop_action)

        # 도움말 메뉴
        self._help_menu = mb.addMenu("도움말(&H)")
        tutorial_action = QAction("튜토리얼(&T)", self)
        tutorial_action.triggered.connect(self._show_tutorial)
        self._help_menu.addAction(tutorial_action)
        self._help_menu.addSeparator()

        doctor_action = QAction("환경 진단(&D)", self)
        doctor_action.triggered.connect(self._show_doctor_report)
        self._help_menu.addAction(doctor_action)

        support_action = QAction("지원 번들 생성(&B)", self)
        support_action.triggered.connect(self._create_support_bundle)
        self._help_menu.addAction(support_action)

        inspect_support_action = QAction("지원 번들 점검(&I)", self)
        inspect_support_action.triggered.connect(self._inspect_support_bundle)
        self._help_menu.addAction(inspect_support_action)

        update_action = QAction("업데이트 확인(&U)", self)
        update_action.triggered.connect(self._check_for_updates)
        self._help_menu.addAction(update_action)
        self._help_menu.addSeparator()

        about_action = QAction("NavierTwin 정보(&A)", self)
        about_action.triggered.connect(self._show_about)
        self._help_menu.addAction(about_action)

    def _refresh_view_menu(self) -> None:
        """현재 탭 목록과 표시 설정을 보기 메뉴에 노출한다."""
        view_menu = getattr(self, "_view_menu", None)
        tabs = getattr(self, "_tabs", None)
        if view_menu is None or tabs is None:
            return

        view_menu.clear()
        for i in range(tabs.count()):
            title = tabs.tabText(i)
            action = QAction(f"{title} 탭", self)
            if i < 9:
                action.setShortcut(f"Ctrl+{i + 1}")
            action.setData(i)
            action.triggered.connect(self._switch_tab)
            view_menu.addAction(action)

        view_menu.addSeparator()
        for theme, label in (("dark", "다크 테마"), ("light", "라이트 테마")):
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(self._config.theme == theme)
            action.setData(theme)
            action.triggered.connect(self._switch_theme)
            view_menu.addAction(action)

        view_menu.addSeparator()
        for lang, label in (("ko", "한국어"), ("en", "English")):
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(self._config.language == lang)
            action.setData(lang)
            action.triggered.connect(self._switch_language)
            view_menu.addAction(action)

    def _setup_statusbar(self) -> None:
        sb: QStatusBar = self.statusBar()

        self._status_label = QLabel("준비")
        sb.addWidget(self._status_label, stretch=1)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFixedWidth(200)
        self._progress_bar.setVisible(False)
        sb.addPermanentWidget(self._progress_bar)

        version_label = QLabel(f"v{__version__}")
        version_label.setObjectName("subtitleLabel")
        sb.addPermanentWidget(version_label)

    def _connect_signals(self) -> None:
        """탭 간 데이터 전달 시그널을 연결한다."""
        # Import → 다른 패널에 dataset 전달
        self._import_panel.dataset_loaded.connect(self._on_dataset_loaded)

        # Reduce → Model 패널에 reducer 전달
        self._reduce_panel.reduction_done.connect(self._on_reduction_done)

        # Model → Twin 패널에 surrogate 전달
        self._model_panel.model_trained.connect(self._on_model_trained)
        self._model_panel.active_learning_done.connect(
            lambda result: self._set_status(
                f"Active Learning 후보 추천 완료: {len(result.get('selected', []))}개"
            )
        )
        self._model_panel.online_learning_done.connect(
            lambda result: self._set_status(
                f"Online Update 완료: buffer={int(result.get('buffer_size', 0))}"
            )
        )

        # Analyze 완료 상태 업데이트
        self._analyze_panel.analysis_done.connect(
            lambda method, _: self._set_status(f"분석 완료 ({method})")
        )

        # Export 완료 상태 업데이트
        self._export_panel.export_done.connect(self._on_export_done)
        self._export_panel.project_loaded.connect(self._on_project_loaded)

        # Twin 예측 완료
        self._twin_panel.prediction_done.connect(
            lambda _: self._set_status("예측 완료")
        )
        self._twin_panel.optimization_done.connect(
            lambda result: self._set_status(
                f"최적화 완료: f_best={float(result.get('f_best', 0.0)):.4g}"
            )
        )
        self._twin_panel.assimilation_done.connect(
            lambda result: self._set_status(
                f"{result.get('method', 'Assimilation')} 완료: "
                f"error={float(result.get('error', 0.0)):.4g}"
            )
        )
        self._twin_panel.design_optimization_done.connect(
            lambda result: self._set_status(
                f"{result.get('method', 'Design Optimization')} 완료"
            )
        )
        if self._explain_panel is not None:
            self._explain_panel.explanation_done.connect(
                lambda result: self._set_status(
                    f"SHAP 설명 완료: {len(result.get('feature_names', []))} features"
                )
            )

        # Simulation 결과 → 상태바 + 전역 viewer 연동 훅
        if self._simulation_panel is not None:
            self._simulation_panel.simulation_done.connect(self._on_simulation_done)

    # ------------------------------------------------------------------
    def _on_simulation_done(self, kind: str, result: object) -> None:
        """SimulationPanel 결과를 Twin 탭 VTK viewer 로 전달 (가능한 경우)."""
        self._set_status(
            f"시뮬레이션 완료: {kind} — {getattr(result, 'get', lambda *_: '')('summary', '')}"
        )
        # Twin 패널에 vtk_viewer 가 있으면 렌더 시도
        viewer = None
        for cand in (self._twin_panel, self._import_panel):
            v = getattr(cand, "_viewer", None) or getattr(cand, "_vtk_viewer", None)
            if v is not None:
                viewer = v
                break
        if viewer is None:
            return
        try:
            if kind == "lbm_cavity":
                snaps = result.get("snapshots")  # (nt, ny, nx, 3)
                if snaps is not None and hasattr(viewer, "load_field_grid_2d"):
                    viewer.load_field_grid_2d(snaps[..., 1], field_name="ux")
            elif kind == "burgers":
                U = result.get("U")
                if U is not None and hasattr(viewer, "load_1d_trajectory"):
                    viewer.load_1d_trajectory(U, field_name="u")
        except Exception:  # noqa: BLE001
            pass

    # ──────────────────────────────────────────────────────────────────
    # 시그널 핸들러
    # ──────────────────────────────────────────────────────────────────

    def _on_dataset_loaded(self, dataset: object) -> None:
        self._latest_dataset = dataset
        self._latest_reducer = None
        self._latest_surrogate = None
        self._latest_operator = None
        self._latest_engine = None
        self._set_status(
            f"데이터셋 로드 완료 — {dataset.n_points} pts, "  # type: ignore[union-attr]
            f"{dataset.n_cells} cells, {dataset.n_time_steps} steps"
        )
        self._analyze_panel.set_dataset(dataset)  # type: ignore[arg-type]
        self._reduce_panel.set_dataset(dataset)    # type: ignore[arg-type]
        self._model_panel.set_dataset(dataset)     # type: ignore[arg-type]
        self._export_panel.set_dataset(dataset)    # type: ignore[arg-type]
        if self._explain_panel is not None:
            self._explain_panel.set_dataset(dataset)
        if self._postproc_panel is not None:
            self._postproc_panel.set_dataset(dataset)
        # Import 탭 완료 후 Analyze 탭으로 자동 이동
        self._tabs.setCurrentIndex(1)

    def _on_reduction_done(self, method: str, reducer: object) -> None:
        self._latest_reducer = reducer
        self._set_status(f"차원 축소 완료 ({method})")
        self._model_panel.set_reducer(reducer)
        artifact = self._reduce_panel.get_reduction_artifact()
        if artifact is not None:
            self._model_panel.set_reduction_artifact(artifact)
        self._tabs.setCurrentIndex(3)

    def _on_model_trained(self, model_type: str, surrogate: object) -> None:
        if self._is_operator_model(model_type, surrogate):
            self._latest_operator = surrogate
            self._export_panel.set_model(surrogate)
            self._set_status(
                f"연산자 학습 완료 ({model_type}) — TwinEngine 자동 연결 생략"
            )
            self._record_model_comparison(model_type, surrogate)
            self._tabs.setCurrentWidget(self._model_panel)
            return

        self._latest_surrogate = surrogate
        if self._explain_panel is not None:
            self._explain_panel.set_model(surrogate)
        if self._latest_reducer is not None:
            try:
                engine = self._build_engine(self._latest_reducer, surrogate)
                self._latest_engine = engine
                self._twin_panel.set_engine(engine)
                self._export_panel.set_engine(engine)
                self._set_status(f"모델 학습 완료 ({model_type}) — TwinEngine 연결 완료")
            except Exception as exc:
                self._set_status(f"모델 학습 완료 ({model_type}) — 엔진 구성 실패: {exc}")
        else:
            self._set_status(f"모델 학습 완료 ({model_type})")
        self._record_model_comparison(model_type, surrogate)
        self._tabs.setCurrentIndex(4)

    def _on_project_loaded(self, dataset: object, engine: object | None) -> None:
        """Export 패널에서 로드한 프로젝트를 전체 워크플로우 상태로 복원한다."""
        self._on_dataset_loaded(dataset)
        load_warning = self._export_panel.last_project_load_warning()
        if engine is not None:
            self._latest_engine = engine
            self._latest_reducer = getattr(engine, "reducer", self._latest_reducer)
            self._latest_surrogate = getattr(engine, "surrogate", self._latest_surrogate)
            if self._latest_reducer is not None:
                self._model_panel.set_reducer(self._latest_reducer)
            self._twin_panel.set_engine(engine)
            self._export_panel.set_engine(engine)
            if self._latest_surrogate is not None and self._explain_panel is not None:
                self._explain_panel.set_model(self._latest_surrogate)
            self._set_status("프로젝트 로드 완료 (dataset + TwinEngine)")
        elif load_warning:
            self._latest_engine = None
            self._set_status(f"프로젝트 부분 로드 완료 (dataset) — {load_warning}")
        else:
            self._latest_engine = None
            self._set_status("프로젝트 로드 완료 (dataset)")
        loaded_meta = self._extract_project_metadata(dataset, engine)
        self._model_panel.set_loaded_metadata(loaded_meta)
        project_path = self._current_project_path()
        if project_path is not None:
            self._remember_recent_project(project_path)

    def _on_export_done(self, path: str) -> None:
        """내보내기 완료 상태를 표시하고 .ntwin 프로젝트를 MRU에 반영한다."""
        self._set_status(f"내보내기 완료: {path}")
        out = Path(path)
        if out.suffix.lower() == ".ntwin":
            self._remember_recent_project(out)

    def _build_engine(self, reducer: object, surrogate: object) -> object:
        """학습된 reducer/surrogate로 TwinEngine을 구성한다."""
        from naviertwin.core.digital_twin.twin_engine import TwinEngine

        return TwinEngine.from_fitted_components(reducer, surrogate)

    @staticmethod
    def _is_operator_model(model_type: str, model: object) -> bool:
        """TwinEngine surrogate 경로와 호환되지 않는 neural operator인지 판정한다."""
        operator_types = {"fno1d", "fno2d", "tfno2d", "deeponet", "unet2d", "wno1d"}
        if model_type.lower() in operator_types:
            return True
        try:
            from naviertwin.core.operator_learning.base import BaseOperator
        except Exception:  # noqa: BLE001
            return False
        return isinstance(model, BaseOperator)

    # ──────────────────────────────────────────────────────────────────
    # 메뉴 액션
    # ──────────────────────────────────────────────────────────────────

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "CFD 파일 열기",
            "",
            open_file_filter(),
        )
        if path:
            self._open_selected_path(Path(path))

    def _open_selected_path(self, path: Path) -> bool:
        """파일 메뉴 선택 경로를 포맷에 맞는 GUI 경로로 전달한다."""
        if path.suffix.lower() == ".ntwin":
            self._tabs.setCurrentWidget(self._export_panel)
            if self._export_panel.load_project_path(path):
                return True
            detail = self._export_panel.last_project_load_error() or "알 수 없는 오류"
            self._set_status("프로젝트 열기 실패")
            QMessageBox.warning(
                self,
                "프로젝트 열기 실패",
                f"{path.name} 파일을 열 수 없습니다.\n\n{detail}",
            )
            return False

        self._import_panel._path_edit.setText(str(path))
        self._tabs.setCurrentWidget(self._import_panel)
        return True

    def _save_project(self) -> None:
        self._tabs.setCurrentIndex(5)
        self._export_panel._save_project()

    def _switch_tab(self) -> None:
        action: QAction = self.sender()  # type: ignore[assignment]
        idx = action.data()
        if isinstance(idx, int):
            self._tabs.setCurrentIndex(idx)

    def _switch_theme(self) -> None:
        action: QAction = self.sender()  # type: ignore[assignment]
        theme = action.data()
        if isinstance(theme, str):
            self.set_theme(theme)
            self._save_gui_config()
            self._set_status(f"테마 변경: {theme}")

    def _switch_language(self) -> None:
        action: QAction = self.sender()  # type: ignore[assignment]
        lang = action.data()
        if isinstance(lang, str):
            self.set_language(lang)
            self._save_gui_config()
            self._set_status(f"언어 변경: {lang}")

    def _run_benchmark(self) -> None:
        """고객 smoke benchmark를 GUI에서 실행한다."""
        try:
            code = self._run_benchmark_cli("burgers")
        except Exception as exc:  # noqa: BLE001
            self._set_status("벤치마크 실패")
            QMessageBox.warning(self, "벤치마크 실패", str(exc))
            return
        if code != 0:
            self._set_status("벤치마크 실패")
            QMessageBox.warning(
                self,
                "벤치마크 실패",
                f"benchmark 종료 코드: {code}",
            )
            return
        self._set_status("벤치마크 완료: burgers")
        QMessageBox.information(
            self,
            "벤치마크 완료",
            "Burgers benchmark가 정상 완료되었습니다.",
        )

    def _run_benchmark_cli(self, kind: str) -> int:
        """테스트에서 대체 가능한 benchmark 실행 래퍼."""
        from naviertwin.main import _run_benchmark

        return _run_benchmark(kind)

    def _run_pipeline_demo(self) -> None:
        outdir = QFileDialog.getExistingDirectory(
            self,
            "파이프라인 데모 출력 폴더 선택",
            "",
        )
        if outdir:
            self._run_pipeline_demo_path(Path(outdir))

    def _run_pipeline_demo_path(self, outdir: Path) -> None:
        """CLI pipeline-demo 워크플로우를 GUI에서 실행한다."""
        try:
            code = self._run_pipeline_demo_cli(outdir)
        except Exception as exc:  # noqa: BLE001
            self._set_status("파이프라인 데모 실패")
            QMessageBox.warning(self, "파이프라인 데모 실패", str(exc))
            return
        if code != 0:
            self._set_status("파이프라인 데모 실패")
            QMessageBox.warning(
                self,
                "파이프라인 데모 실패",
                f"pipeline-demo 종료 코드: {code}",
            )
            return
        self._set_status("파이프라인 데모 완료")
        QMessageBox.information(
            self,
            "파이프라인 데모 완료",
            f"metrics.json 및 report.html 생성 위치:\n{outdir}",
        )

    def _run_pipeline_demo_cli(self, outdir: Path) -> int:
        """테스트에서 대체 가능한 pipeline-demo 실행 래퍼."""
        from naviertwin.main import _run_pipeline_demo

        return _run_pipeline_demo(outdir=str(outdir), n_modes=3, surrogate="rbf")

    def _build_twin_from_csv_snapshots(self) -> None:
        """CSV 스냅샷 시퀀스에서 고객용 TwinEngine 산출물을 생성한다."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "CSV 스냅샷 선택",
            "",
            "CSV snapshots (*.csv)",
        )
        if not paths:
            return

        field_column, ok = QInputDialog.getText(
            self,
            "필드 컬럼",
            "학습할 scalar/vector 성분 컬럼명:",
            text="U",
        )
        field_column = field_column.strip()
        if not ok or not field_column:
            return

        outdir = QFileDialog.getExistingDirectory(
            self,
            "트윈 산출물 저장 폴더 선택",
            "",
        )
        if outdir:
            self._build_twin_from_csv_paths(
                [Path(path) for path in paths],
                field_column=field_column,
                outdir=Path(outdir),
            )

    def _build_twin_from_csv_paths(
        self,
        csv_paths: list[Path],
        *,
        field_column: str,
        outdir: Path,
    ) -> None:
        """GUI에서 build-twin CLI 워크플로우를 실행하고 결과 엔진을 연결한다."""
        try:
            code = self._run_build_twin_cli(csv_paths, field_column=field_column, outdir=outdir)
        except Exception as exc:  # noqa: BLE001
            self._set_status("트윈 생성 실패")
            QMessageBox.warning(self, "트윈 생성 실패", str(exc))
            return
        if code != 0:
            self._set_status("트윈 생성 실패")
            QMessageBox.warning(
                self,
                "트윈 생성 실패",
                f"build-twin 종료 코드: {code}",
            )
            return

        engine_path = outdir / "engine.pkl"
        if engine_path.exists():
            self._load_engine_artifact(engine_path)
        self._set_status("트윈 생성 완료")
        QMessageBox.information(
            self,
            "트윈 생성 완료",
            f"engine.pkl, metrics.json, report.html 생성 위치:\n{outdir}",
        )

    def _run_build_twin_cli(
        self,
        csv_paths: list[Path],
        *,
        field_column: str,
        outdir: Path,
    ) -> int:
        """테스트에서 대체 가능한 build-twin 실행 래퍼."""
        from naviertwin.main import _run_build_twin

        return _run_build_twin(
            input_path=None,
            csv_snapshots=",".join(str(path) for path in csv_paths),
            field=None,
            field_column=field_column,
            params=None,
            param_columns=None,
            outdir=str(outdir),
            reducer="pod",
            n_modes=3,
            surrogate="rbf",
            validation_count=3,
            as_json=False,
        )

    def _predict_twin_from_engine(self) -> None:
        """저장된 TwinEngine 아티팩트로 파라미터 예측을 실행한다."""
        engine_path, _ = QFileDialog.getOpenFileName(
            self,
            "TwinEngine 선택",
            "",
            "Pickle (*.pkl)",
        )
        if not engine_path:
            return

        params, ok = QInputDialog.getText(
            self,
            "예측 파라미터",
            "쉼표 구분 파라미터 값:",
            text="0.5",
        )
        params = params.strip()
        if not ok or not params:
            return

        output, _ = QFileDialog.getSaveFileName(
            self,
            "예측 CSV 저장",
            "prediction.csv",
            "CSV (*.csv)",
        )
        self._predict_twin_from_engine_path(
            Path(engine_path),
            params=params,
            output=Path(output) if output else None,
        )

    def _predict_twin_from_engine_path(
        self,
        engine_path: Path,
        *,
        params: str,
        output: Path | None,
    ) -> None:
        """GUI에서 predict-twin CLI 워크플로우를 실행한다."""
        try:
            code = self._run_predict_twin_cli(engine_path, params=params, output=output)
        except Exception as exc:  # noqa: BLE001
            self._set_status("트윈 예측 실패")
            QMessageBox.warning(self, "트윈 예측 실패", str(exc))
            return
        if code != 0:
            self._set_status("트윈 예측 실패")
            QMessageBox.warning(
                self,
                "트윈 예측 실패",
                f"predict-twin 종료 코드: {code}",
            )
            return

        if engine_path.exists():
            self._load_engine_artifact(engine_path)
        self._set_status("트윈 예측 완료")
        suffix = f"\n저장 위치: {output}" if output is not None else ""
        QMessageBox.information(
            self,
            "트윈 예측 완료",
            f"저장된 TwinEngine 예측이 완료되었습니다.{suffix}",
        )

    def _predict_twin_from_artifacts_dir(self) -> None:
        """배포/추출된 트윈 산출물 디렉토리로 파라미터 예측을 실행한다."""
        artifacts_dir = QFileDialog.getExistingDirectory(
            self,
            "트윈 산출물 디렉토리 선택",
            "",
        )
        if not artifacts_dir:
            return

        params, ok = QInputDialog.getText(
            self,
            "예측 파라미터",
            "쉼표 구분 파라미터 값:",
            text="0.5",
        )
        params = params.strip()
        if not ok or not params:
            return

        output, _ = QFileDialog.getSaveFileName(
            self,
            "예측 CSV 저장",
            "prediction.csv",
            "CSV (*.csv)",
        )
        self._predict_twin_from_artifacts_dir_path(
            Path(artifacts_dir),
            params=params,
            output=Path(output) if output else None,
        )

    def _predict_twin_from_artifacts_dir_path(
        self,
        artifacts_dir: Path,
        *,
        params: str,
        output: Path | None,
    ) -> None:
        """GUI에서 predict-twin --artifacts-dir 워크플로우를 실행한다."""
        try:
            code = self._run_predict_twin_artifacts_cli(
                artifacts_dir,
                params=params,
                output=output,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status("배포 트윈 예측 실패")
            QMessageBox.warning(self, "배포 트윈 예측 실패", str(exc))
            return
        if code != 0:
            self._set_status("배포 트윈 예측 실패")
            QMessageBox.warning(
                self,
                "배포 트윈 예측 실패",
                f"predict-twin 종료 코드: {code}",
            )
            return

        engine_path = artifacts_dir / "engine.pkl"
        if engine_path.exists():
            self._load_engine_artifact(engine_path)
        self._set_status("배포 트윈 예측 완료")
        suffix = f"\n저장 위치: {output}" if output is not None else ""
        QMessageBox.information(
            self,
            "배포 트윈 예측 완료",
            f"배포된 트윈 디렉토리 예측이 완료되었습니다.{suffix}",
        )

    def _run_predict_twin_cli(
        self,
        engine_path: Path,
        *,
        params: str,
        output: Path | None,
    ) -> int:
        """테스트에서 대체 가능한 predict-twin 실행 래퍼."""
        from naviertwin.main import _run_predict_twin

        return _run_predict_twin(
            engine_path=str(engine_path),
            artifacts_dir=None,
            params=params,
            params_csv=None,
            param_columns=None,
            output=str(output) if output is not None else None,
            as_json=False,
        )

    def _run_predict_twin_artifacts_cli(
        self,
        artifacts_dir: Path,
        *,
        params: str,
        output: Path | None,
    ) -> int:
        """테스트에서 대체 가능한 predict-twin --artifacts-dir 실행 래퍼."""
        from naviertwin.main import _run_predict_twin

        return _run_predict_twin(
            engine_path=None,
            artifacts_dir=str(artifacts_dir),
            params=params,
            params_csv=None,
            param_columns=None,
            output=str(output) if output is not None else None,
            as_json=False,
        )

    def _benchmark_twin_from_artifacts_dir(self) -> None:
        """배포/추출된 트윈 산출물 디렉토리의 예측 latency를 측정한다."""
        artifacts_dir = QFileDialog.getExistingDirectory(
            self,
            "트윈 산출물 디렉토리 선택",
            "",
        )
        if not artifacts_dir:
            return

        params: str | None = None
        params_csv: Path | None = None
        use_csv = (
            QMessageBox.question(
                self,
                "배치 파라미터",
                "CSV 파일의 numeric 컬럼 전체로 배치 latency를 측정하시겠습니까?",
            )
            == QMessageBox.StandardButton.Yes
        )
        if use_csv:
            selected_csv, _ = QFileDialog.getOpenFileName(
                self,
                "파라미터 CSV 선택",
                "",
                "CSV (*.csv)",
            )
            if not selected_csv:
                return
            params_csv = Path(selected_csv)
        else:
            raw_params, ok = QInputDialog.getText(
                self,
                "벤치마크 파라미터",
                "쉼표 구분 파라미터 값:",
                text="0.5",
            )
            params = raw_params.strip()
            if not ok or not params:
                return

        counts, ok = QInputDialog.getText(
            self,
            "벤치마크 반복",
            "warmup,repeat:",
            text="2,20",
        )
        if not ok:
            return
        try:
            warmup, repeat = self._parse_benchmark_counts(counts)
        except ValueError as exc:
            self._set_status("배포 트윈 지연시간 측정 실패")
            QMessageBox.warning(self, "배포 트윈 지연시간 측정 실패", str(exc))
            return

        slo_text, ok = QInputDialog.getText(
            self,
            "벤치마크 SLO",
            "max_p95_ms,min_throughput_hz (비우면 게이트 없음):",
            text="",
        )
        if not ok:
            return
        try:
            max_p95_ms, min_throughput_hz = self._parse_benchmark_slo(slo_text)
        except ValueError as exc:
            self._set_status("배포 트윈 지연시간 측정 실패")
            QMessageBox.warning(self, "배포 트윈 지연시간 측정 실패", str(exc))
            return

        output, _ = QFileDialog.getSaveFileName(
            self,
            "지연시간 JSON 저장",
            "latency.json",
            "JSON (*.json)",
        )
        self._benchmark_twin_from_artifacts_dir_path(
            Path(artifacts_dir),
            params=params,
            params_csv=params_csv,
            warmup=warmup,
            repeat=repeat,
            max_p95_ms=max_p95_ms,
            min_throughput_hz=min_throughput_hz,
            output=Path(output) if output else None,
        )

    def _benchmark_twin_from_artifacts_dir_path(
        self,
        artifacts_dir: Path,
        *,
        params: str | None = None,
        params_csv: Path | None = None,
        warmup: int,
        repeat: int,
        max_p95_ms: float | None = None,
        min_throughput_hz: float | None = None,
        output: Path | None,
    ) -> None:
        """GUI에서 benchmark-twin --artifacts-dir 워크플로우를 실행한다."""
        try:
            code = self._run_benchmark_twin_artifacts_cli(
                artifacts_dir,
                params=params,
                params_csv=params_csv,
                warmup=warmup,
                repeat=repeat,
                max_p95_ms=max_p95_ms,
                min_throughput_hz=min_throughput_hz,
                output=output,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status("배포 트윈 지연시간 측정 실패")
            QMessageBox.warning(self, "배포 트윈 지연시간 측정 실패", str(exc))
            return
        if code != 0:
            self._set_status("배포 트윈 지연시간 측정 실패")
            QMessageBox.warning(
                self,
                "배포 트윈 지연시간 측정 실패",
                f"benchmark-twin 종료 코드: {code}",
            )
            return

        self._set_status("배포 트윈 지연시간 측정 완료")
        suffix = f"\n저장 위치: {output}" if output is not None else ""
        QMessageBox.information(
            self,
            "배포 트윈 지연시간 측정 완료",
            f"배포된 트윈 디렉토리 지연시간 측정이 완료되었습니다.{suffix}",
        )

    def _run_benchmark_twin_artifacts_cli(
        self,
        artifacts_dir: Path,
        *,
        params: str | None = None,
        params_csv: Path | None = None,
        warmup: int,
        repeat: int,
        max_p95_ms: float | None = None,
        min_throughput_hz: float | None = None,
        output: Path | None,
    ) -> int:
        """테스트에서 대체 가능한 benchmark-twin --artifacts-dir 실행 래퍼."""
        from naviertwin.main import _run_benchmark_twin

        return _run_benchmark_twin(
            engine_path=None,
            artifacts_dir=str(artifacts_dir),
            params=params,
            params_csv=str(params_csv) if params_csv is not None else None,
            param_columns=None,
            warmup=warmup,
            repeat=repeat,
            max_mean_ms=None,
            max_p50_ms=None,
            max_p95_ms=max_p95_ms,
            max_p99_ms=None,
            min_throughput_hz=min_throughput_hz,
            output=str(output) if output is not None else None,
            as_json=False,
        )

    def _validate_twin_from_engine(self) -> None:
        """저장된 TwinEngine을 선택한 CSV 기준 snapshot과 비교 검증한다."""
        engine_path, _ = QFileDialog.getOpenFileName(
            self,
            "TwinEngine 선택",
            "",
            "Pickle (*.pkl)",
        )
        if not engine_path:
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "검증 CSV 스냅샷 선택",
            "",
            "CSV snapshots (*.csv)",
        )
        if not paths:
            return

        field_column, ok = QInputDialog.getText(
            self,
            "필드 컬럼",
            "검증할 scalar/vector 성분 컬럼명:",
            text="U",
        )
        field_column = field_column.strip()
        if not ok or not field_column:
            return

        thresholds, ok = QInputDialog.getText(
            self,
            "검증 기준",
            "max_rmse,min_r2,max_relative_l2 (빈 값 허용):",
            text="",
        )
        if not ok:
            return
        try:
            max_rmse, min_r2, max_relative_l2 = self._parse_validation_thresholds(
                thresholds
            )
        except ValueError as exc:
            self._set_status("트윈 검증 실패")
            QMessageBox.warning(self, "트윈 검증 실패", str(exc))
            return

        output, _ = QFileDialog.getSaveFileName(
            self,
            "검증 JSON 저장",
            "validation.json",
            "JSON (*.json)",
        )
        self._validate_twin_from_paths(
            Path(engine_path),
            [Path(path) for path in paths],
            field_column=field_column,
            output=Path(output) if output else None,
            max_rmse=max_rmse,
            min_r2=min_r2,
            max_relative_l2=max_relative_l2,
        )

    def _validate_twin_from_artifacts_dir(self) -> None:
        """배포/추출된 트윈 산출물 디렉토리를 CSV 기준 snapshot으로 검증한다."""
        artifacts_dir = QFileDialog.getExistingDirectory(
            self,
            "트윈 산출물 디렉토리 선택",
            "",
        )
        if not artifacts_dir:
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "검증 CSV 스냅샷 선택",
            "",
            "CSV snapshots (*.csv)",
        )
        if not paths:
            return

        field_column, ok = QInputDialog.getText(
            self,
            "필드 컬럼",
            "검증할 scalar/vector 성분 컬럼명:",
            text="U",
        )
        field_column = field_column.strip()
        if not ok or not field_column:
            return

        thresholds, ok = QInputDialog.getText(
            self,
            "검증 기준",
            "max_rmse,min_r2,max_relative_l2 (빈 값 허용):",
            text="",
        )
        if not ok:
            return
        try:
            max_rmse, min_r2, max_relative_l2 = self._parse_validation_thresholds(
                thresholds
            )
        except ValueError as exc:
            self._set_status("배포 트윈 검증 실패")
            QMessageBox.warning(self, "배포 트윈 검증 실패", str(exc))
            return

        output, _ = QFileDialog.getSaveFileName(
            self,
            "검증 JSON 저장",
            "validation.json",
            "JSON (*.json)",
        )
        self._validate_twin_from_artifacts_dir_paths(
            Path(artifacts_dir),
            [Path(path) for path in paths],
            field_column=field_column,
            output=Path(output) if output else None,
            max_rmse=max_rmse,
            min_r2=min_r2,
            max_relative_l2=max_relative_l2,
        )

    def _validate_twin_from_paths(
        self,
        engine_path: Path,
        csv_paths: list[Path],
        *,
        field_column: str,
        output: Path | None,
        max_rmse: float | None = None,
        min_r2: float | None = None,
        max_relative_l2: float | None = None,
    ) -> None:
        """GUI에서 validate-twin CLI 워크플로우를 실행한다."""
        try:
            code = self._run_validate_twin_cli(
                engine_path,
                csv_paths,
                field_column=field_column,
                output=output,
                max_rmse=max_rmse,
                min_r2=min_r2,
                max_relative_l2=max_relative_l2,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status("트윈 검증 실패")
            QMessageBox.warning(self, "트윈 검증 실패", str(exc))
            return
        if code != 0:
            self._set_status("트윈 검증 실패")
            QMessageBox.warning(
                self,
                "트윈 검증 실패",
                f"validate-twin 종료 코드: {code}",
            )
            return

        if engine_path.exists():
            self._load_engine_artifact(engine_path)
        self._set_status("트윈 검증 완료")
        suffix = f"\n저장 위치: {output}" if output is not None else ""
        QMessageBox.information(
            self,
            "트윈 검증 완료",
            f"저장된 TwinEngine 검증이 완료되었습니다.{suffix}",
        )

    def _validate_twin_from_artifacts_dir_paths(
        self,
        artifacts_dir: Path,
        csv_paths: list[Path],
        *,
        field_column: str,
        output: Path | None,
        max_rmse: float | None = None,
        min_r2: float | None = None,
        max_relative_l2: float | None = None,
    ) -> None:
        """GUI에서 validate-twin --artifacts-dir 워크플로우를 실행한다."""
        try:
            code = self._run_validate_twin_artifacts_cli(
                artifacts_dir,
                csv_paths,
                field_column=field_column,
                output=output,
                max_rmse=max_rmse,
                min_r2=min_r2,
                max_relative_l2=max_relative_l2,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status("배포 트윈 검증 실패")
            QMessageBox.warning(self, "배포 트윈 검증 실패", str(exc))
            return
        if code != 0:
            self._set_status("배포 트윈 검증 실패")
            QMessageBox.warning(
                self,
                "배포 트윈 검증 실패",
                f"validate-twin 종료 코드: {code}",
            )
            return

        engine_path = artifacts_dir / "engine.pkl"
        if engine_path.exists():
            self._load_engine_artifact(engine_path)
        self._set_status("배포 트윈 검증 완료")
        suffix = f"\n저장 위치: {output}" if output is not None else ""
        QMessageBox.information(
            self,
            "배포 트윈 검증 완료",
            f"배포된 트윈 디렉토리 검증이 완료되었습니다.{suffix}",
        )

    def _run_validate_twin_cli(
        self,
        engine_path: Path,
        csv_paths: list[Path],
        *,
        field_column: str,
        output: Path | None,
        max_rmse: float | None = None,
        min_r2: float | None = None,
        max_relative_l2: float | None = None,
    ) -> int:
        """테스트에서 대체 가능한 validate-twin 실행 래퍼."""
        from naviertwin.main import _run_validate_twin

        return _run_validate_twin(
            engine_path=str(engine_path),
            artifacts_dir=None,
            input_path=None,
            csv_snapshots=",".join(str(path) for path in csv_paths),
            field=None,
            field_column=field_column,
            params=None,
            param_columns=None,
            max_rmse=max_rmse,
            min_r2=min_r2,
            max_relative_l2=max_relative_l2,
            output=str(output) if output is not None else None,
            as_json=False,
        )

    def _run_validate_twin_artifacts_cli(
        self,
        artifacts_dir: Path,
        csv_paths: list[Path],
        *,
        field_column: str,
        output: Path | None,
        max_rmse: float | None = None,
        min_r2: float | None = None,
        max_relative_l2: float | None = None,
    ) -> int:
        """테스트에서 대체 가능한 validate-twin --artifacts-dir 실행 래퍼."""
        from naviertwin.main import _run_validate_twin

        return _run_validate_twin(
            engine_path=None,
            artifacts_dir=str(artifacts_dir),
            input_path=None,
            csv_snapshots=",".join(str(path) for path in csv_paths),
            field=None,
            field_column=field_column,
            params=None,
            param_columns=None,
            max_rmse=max_rmse,
            min_r2=min_r2,
            max_relative_l2=max_relative_l2,
            output=str(output) if output is not None else None,
            as_json=False,
        )

    @staticmethod
    def _parse_validation_thresholds(
        value: str,
    ) -> tuple[float | None, float | None, float | None]:
        """GUI threshold 입력 문자열을 validate-twin 인자로 변환한다."""
        stripped = value.strip()
        if not stripped:
            return None, None, None
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) > 3:
            raise ValueError("검증 기준은 max_rmse,min_r2,max_relative_l2 순서로 최대 3개입니다.")
        parsed: list[float | None] = []
        for part in parts:
            if not part:
                parsed.append(None)
                continue
            try:
                parsed.append(float(part))
            except ValueError as exc:
                raise ValueError(f"검증 기준은 숫자여야 합니다: {part}") from exc
        while len(parsed) < 3:
            parsed.append(None)
        return parsed[0], parsed[1], parsed[2]

    @staticmethod
    def _parse_benchmark_counts(value: str) -> tuple[int, int]:
        """GUI benchmark 입력 문자열을 warmup/repeat 정수로 변환한다."""
        stripped = value.strip()
        if not stripped:
            return 2, 20
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) > 2:
            raise ValueError("벤치마크 반복은 warmup,repeat 순서로 최대 2개입니다.")
        warmup = int(parts[0]) if parts and parts[0] else 2
        repeat = int(parts[1]) if len(parts) > 1 and parts[1] else 20
        if warmup < 0:
            raise ValueError("warmup은 0 이상이어야 합니다.")
        if repeat < 1:
            raise ValueError("repeat는 1 이상이어야 합니다.")
        return warmup, repeat

    @staticmethod
    def _parse_benchmark_slo(value: str) -> tuple[float | None, float | None]:
        """GUI benchmark SLO 입력 문자열을 p95/throughput 게이트로 변환한다."""
        stripped = value.strip()
        if not stripped:
            return None, None
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) > 2:
            raise ValueError("SLO는 max_p95_ms,min_throughput_hz 순서로 최대 2개입니다.")

        parsed: list[float | None] = []
        for part in parts:
            if not part:
                parsed.append(None)
                continue
            try:
                number = float(part)
            except ValueError as exc:
                raise ValueError(f"SLO 값은 숫자여야 합니다: {part}") from exc
            if number <= 0:
                raise ValueError("SLO 값은 양수여야 합니다.")
            parsed.append(number)
        while len(parsed) < 2:
            parsed.append(None)
        return parsed[0], parsed[1]

    def _package_twin_artifacts(self) -> None:
        """build-twin 산출물 디렉토리를 고객 전달용 ZIP으로 패키징한다."""
        artifacts_dir = QFileDialog.getExistingDirectory(
            self,
            "트윈 산출물 폴더 선택",
            "",
        )
        if not artifacts_dir:
            return

        include_validation: Path | None = None
        if (
            QMessageBox.question(
                self,
                "검증 리포트 포함",
                "별도 validation JSON 리포트를 ZIP에 포함하시겠습니까?",
            )
            == QMessageBox.StandardButton.Yes
        ):
            validation, _ = QFileDialog.getOpenFileName(
                self,
                "검증 JSON 선택",
                "",
                "JSON (*.json)",
            )
            if validation:
                include_validation = Path(validation)

        output, _ = QFileDialog.getSaveFileName(
            self,
            "트윈 ZIP 저장",
            "naviertwin-delivery.zip",
            "ZIP (*.zip)",
        )
        if output:
            self._package_twin_from_paths(
                Path(artifacts_dir),
                output=Path(output),
                include_validation=include_validation,
            )

    def _package_twin_from_paths(
        self,
        artifacts_dir: Path,
        *,
        output: Path,
        include_validation: Path | None = None,
    ) -> None:
        """GUI에서 package-twin CLI 워크플로우를 실행한다."""
        try:
            code = self._run_package_twin_cli(
                artifacts_dir,
                output=output,
                include_validation=include_validation,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status("트윈 패키징 실패")
            QMessageBox.warning(self, "트윈 패키징 실패", str(exc))
            return
        if code != 0:
            self._set_status("트윈 패키징 실패")
            QMessageBox.warning(
                self,
                "트윈 패키징 실패",
                f"package-twin 종료 코드: {code}",
            )
            return

        self._set_status("트윈 패키징 완료")
        QMessageBox.information(
            self,
            "트윈 패키징 완료",
            f"고객 전달용 ZIP 생성 위치:\n{output}",
        )

    def _run_package_twin_cli(
        self,
        artifacts_dir: Path,
        *,
        output: Path,
        include_validation: Path | None = None,
    ) -> int:
        """테스트에서 대체 가능한 package-twin 실행 래퍼."""
        from naviertwin.main import _run_package_twin

        return _run_package_twin(
            artifacts_dir=str(artifacts_dir),
            include_validation=str(include_validation) if include_validation is not None else None,
            output=str(output),
            as_json=False,
        )

    def _verify_twin_package(self) -> None:
        """고객 전달용 트윈 ZIP의 MANIFEST.json 무결성을 검증한다."""
        package_path, _ = QFileDialog.getOpenFileName(
            self,
            "트윈 ZIP 선택",
            "",
            "ZIP (*.zip)",
        )
        if package_path:
            self._verify_twin_package_path(Path(package_path))

    def _inspect_twin_package(self) -> None:
        """고객 전달용 트윈 ZIP의 구성과 메타데이터를 조회한다."""
        package_path, _ = QFileDialog.getOpenFileName(
            self,
            "트윈 ZIP 선택",
            "",
            "ZIP (*.zip)",
        )
        if package_path:
            self._inspect_twin_package_path(Path(package_path))

    def _inspect_twin_package_path(self, package_path: Path) -> None:
        """GUI에서 inspect-twin-package 조회 결과를 요약 표시한다."""
        try:
            payload = self._run_inspect_twin_package_cli(package_path)
        except Exception as exc:  # noqa: BLE001
            self._set_status("트윈 패키지 정보 조회 실패")
            QMessageBox.warning(self, "트윈 패키지 정보 조회 실패", str(exc))
            return
        if payload.get("status") != "ok":
            self._set_status("트윈 패키지 정보 조회 실패")
            errors = payload.get("errors") or []
            error_text = "; ".join(map(str, errors)) if errors else "status != ok"
            QMessageBox.warning(self, "트윈 패키지 정보 조회 실패", error_text)
            return

        metrics = payload.get("metrics") or {}
        rmse = metrics.get("rmse", "-") if isinstance(metrics, dict) else "-"
        r2 = metrics.get("r2", "-") if isinstance(metrics, dict) else "-"
        self._set_status("트윈 패키지 정보 조회 완료")
        QMessageBox.information(
            self,
            "트윈 패키지 정보 조회 완료",
            (
                f"패키지: {package_path}\n"
                f"형식: {payload.get('format') or '-'}\n"
                f"엔트리 수: {payload.get('manifest_entry_count')}\n"
                f"validation 포함: {payload.get('validation_included')}\n"
                f"RMSE: {rmse}\n"
                f"R²: {r2}"
            ),
        )

    def _run_inspect_twin_package_cli(self, package_path: Path) -> dict[str, object]:
        """테스트에서 대체 가능한 inspect-twin-package 실행 래퍼."""
        from naviertwin.main import _inspect_twin_package_archive

        return _inspect_twin_package_archive(package_path)

    def _verify_and_extract_twin_package(self) -> None:
        """고객 전달용 트윈 ZIP을 검증한 뒤 선택한 디렉토리에 추출한다."""
        package_path, _ = QFileDialog.getOpenFileName(
            self,
            "트윈 ZIP 선택",
            "",
            "ZIP (*.zip)",
        )
        if not package_path:
            return
        extract_dir = QFileDialog.getExistingDirectory(
            self,
            "트윈 패키지 추출 디렉토리 선택",
            "",
        )
        if extract_dir:
            self._verify_twin_package_path(Path(package_path), extract_to=Path(extract_dir))

    def _verify_twin_package_path(
        self,
        package_path: Path,
        *,
        extract_to: Path | None = None,
    ) -> None:
        """GUI에서 verify-twin-package CLI 워크플로우를 실행한다."""
        try:
            if extract_to is None:
                code = self._run_verify_twin_package_cli(package_path)
            else:
                code = self._run_verify_twin_package_cli(package_path, extract_to=extract_to)
        except Exception as exc:  # noqa: BLE001
            self._set_status("트윈 패키지 검증 실패")
            QMessageBox.warning(self, "트윈 패키지 검증 실패", str(exc))
            return
        if code != 0:
            self._set_status("트윈 패키지 검증 실패")
            QMessageBox.warning(
                self,
                "트윈 패키지 검증 실패",
                f"verify-twin-package 종료 코드: {code}",
            )
            return

        if extract_to is None:
            self._set_status("트윈 패키지 검증 완료")
            message = f"ZIP 무결성 검증 완료:\n{package_path}"
        else:
            self._set_status("트윈 패키지 검증 및 추출 완료")
            message = f"ZIP 무결성 검증 및 추출 완료:\n{package_path}\n\n추출 위치:\n{extract_to}"
        QMessageBox.information(
            self,
            self._status_label.text(),
            message,
        )

    def _run_verify_twin_package_cli(
        self,
        package_path: Path,
        *,
        extract_to: Path | None = None,
    ) -> int:
        """테스트에서 대체 가능한 verify-twin-package 실행 래퍼."""
        from naviertwin.main import _run_verify_twin_package

        return _run_verify_twin_package(
            package_path=str(package_path),
            extract_to=str(extract_to) if extract_to is not None else None,
            as_json=False,
        )

    def _accept_twin_package(self) -> None:
        """고객 전달용 트윈 ZIP을 검증부터 latency SLO까지 원샷 검사한다."""
        package_path, _ = QFileDialog.getOpenFileName(
            self,
            "트윈 ZIP 선택",
            "",
            "ZIP (*.zip)",
        )
        if not package_path:
            return

        extract_dir = QFileDialog.getExistingDirectory(
            self,
            "트윈 패키지 추출 디렉토리 선택",
            "",
        )
        if not extract_dir:
            return

        counts, ok = QInputDialog.getText(
            self,
            "수락 검사 반복",
            "warmup,repeat:",
            text="2,20",
        )
        if not ok:
            return
        try:
            warmup, repeat = self._parse_benchmark_counts(counts)
        except ValueError as exc:
            self._set_status("트윈 패키지 수락 검사 실패")
            QMessageBox.warning(self, "트윈 패키지 수락 검사 실패", str(exc))
            return

        slo_text, ok = QInputDialog.getText(
            self,
            "수락 검사 SLO",
            "max_p95_ms,min_throughput_hz (비우면 delivery.json latency_slo 사용):",
            text="",
        )
        if not ok:
            return
        try:
            max_p95_ms, min_throughput_hz = self._parse_benchmark_slo(slo_text)
        except ValueError as exc:
            self._set_status("트윈 패키지 수락 검사 실패")
            QMessageBox.warning(self, "트윈 패키지 수락 검사 실패", str(exc))
            return

        output, _ = QFileDialog.getSaveFileName(
            self,
            "수락 검사 JSON 저장",
            "acceptance.json",
            "JSON (*.json)",
        )
        self._accept_twin_package_path(
            Path(package_path),
            extract_to=Path(extract_dir),
            warmup=warmup,
            repeat=repeat,
            max_p95_ms=max_p95_ms,
            min_throughput_hz=min_throughput_hz,
            output=Path(output) if output else None,
            summary_output=Path(output).with_suffix(".md") if output else None,
        )

    def _accept_twin_package_path(
        self,
        package_path: Path,
        *,
        extract_to: Path,
        warmup: int,
        repeat: int,
        max_p95_ms: float | None = None,
        min_throughput_hz: float | None = None,
        output: Path | None,
        summary_output: Path | None = None,
    ) -> None:
        """GUI에서 accept-twin-package CLI 워크플로우를 실행한다."""
        try:
            code = self._run_accept_twin_package_cli(
                package_path,
                extract_to=extract_to,
                warmup=warmup,
                repeat=repeat,
                max_p95_ms=max_p95_ms,
                min_throughput_hz=min_throughput_hz,
                output=output,
                summary_output=summary_output,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status("트윈 패키지 수락 검사 실패")
            QMessageBox.warning(self, "트윈 패키지 수락 검사 실패", str(exc))
            return
        if code != 0:
            self._set_status("트윈 패키지 수락 검사 실패")
            QMessageBox.warning(
                self,
                "트윈 패키지 수락 검사 실패",
                f"accept-twin-package 종료 코드: {code}",
            )
            return

        self._last_acceptance_json = output if output is not None and output.exists() else None
        self._last_acceptance_summary = (
            summary_output if summary_output is not None and summary_output.exists() else None
        )
        self._set_status("트윈 패키지 수락 검사 완료")
        suffix = ""
        if output is not None:
            suffix += f"\nJSON 저장 위치: {output}"
        if summary_output is not None:
            suffix += f"\nMarkdown 요약 위치: {summary_output}"
        QMessageBox.information(
            self,
            "트윈 패키지 수락 검사 완료",
            (
                f"패키지 검증, 샘플 예측, latency SLO 검사가 완료되었습니다.\n"
                f"패키지: {package_path}\n"
                f"추출 위치: {extract_to}{suffix}"
            ),
        )

    def _run_accept_twin_package_cli(
        self,
        package_path: Path,
        *,
        extract_to: Path,
        warmup: int,
        repeat: int,
        max_p95_ms: float | None = None,
        min_throughput_hz: float | None = None,
        output: Path | None,
        summary_output: Path | None = None,
    ) -> int:
        """테스트에서 대체 가능한 accept-twin-package 실행 래퍼."""
        from naviertwin.main import _run_accept_twin_package

        return _run_accept_twin_package(
            package_path=str(package_path),
            extract_to=str(extract_to),
            prediction_output=None,
            warmup=warmup,
            repeat=repeat,
            max_mean_ms=None,
            max_p50_ms=None,
            max_p95_ms=max_p95_ms,
            max_p99_ms=None,
            min_throughput_hz=min_throughput_hz,
            skip_benchmark=False,
            output=str(output) if output is not None else None,
            summary_output=str(summary_output) if summary_output is not None else None,
            as_json=False,
        )

    def _load_engine_artifact(self, engine_path: Path) -> bool:
        """저장된 TwinEngine 아티팩트를 Twin/Export 패널에 연결한다."""
        try:
            from naviertwin.core.digital_twin.twin_engine import TwinEngine

            engine = TwinEngine.load(engine_path)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"TwinEngine 로드 실패: {exc}")
            return False

        self._latest_engine = engine
        self._latest_reducer = getattr(engine, "reducer", self._latest_reducer)
        self._latest_surrogate = getattr(engine, "surrogate", self._latest_surrogate)
        self._twin_panel.set_engine(engine)
        self._export_panel.set_engine(engine)
        return True

    def _start_api_server(self) -> None:
        """FastAPI 서버를 GUI에서 백그라운드 프로세스로 시작한다."""
        if (
            self._server_process is not None
            and self._server_process.state() != QProcess.ProcessState.NotRunning
        ):
            self._set_status("API 서버 이미 실행 중")
            QMessageBox.information(self, "API 서버", "API 서버가 이미 실행 중입니다.")
            return

        process = self._create_api_server_process()
        process.setProgram(sys.executable)
        process.setArguments([
            "-m",
            "naviertwin.main",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ])
        self._server_process = process
        process.finished.connect(self._on_api_server_finished)
        process.start()
        if not process.waitForStarted(1000):
            self._set_status("API 서버 시작 실패")
            QMessageBox.warning(self, "API 서버 시작 실패", "서버 프로세스를 시작하지 못했습니다.")
            return

        self._set_status("API 서버 실행 중: http://127.0.0.1:8000")
        QMessageBox.information(
            self,
            "API 서버 시작",
            "API 서버를 시작했습니다.\nURL: http://127.0.0.1:8000",
        )

    def _stop_api_server(self) -> None:
        """GUI에서 시작한 API 서버 프로세스를 중지한다."""
        process = self._server_process
        if process is None or process.state() == QProcess.ProcessState.NotRunning:
            self._set_status("API 서버 실행 중 아님")
            return
        process.terminate()
        if not process.waitForFinished(1000):
            process.kill()
            process.waitForFinished(1000)
        self._server_process = None
        self._set_status("API 서버 중지됨")

    def _create_api_server_process(self) -> QProcess:
        """테스트에서 대체 가능한 API 서버 프로세스 팩토리."""
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        return process

    def _on_api_server_finished(
        self,
        exit_code: int,
        _exit_status: QProcess.ExitStatus,
    ) -> None:
        """API 서버 프로세스 종료 상태를 GUI에 반영한다."""
        self._server_process = None
        if exit_code == 0:
            self._set_status("API 서버 종료됨")
        else:
            self._set_status(f"API 서버 종료됨: exit={exit_code}")

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "NavierTwin 정보",
            f"<h3>NavierTwin v{__version__}</h3>"
            "<p>CFD 후처리 결과를 AI/ROM 디지털 트윈으로 변환하는 오픈소스 툴.</p>"
            "<p>License: MIT</p>"
            "<p>Python 3.10+ | PySide6 | PyVista | PyTorch</p>",
        )

    def _show_tutorial(self) -> None:
        """신규 사용자 튜토리얼 위저드를 실행한다."""
        wizard = self._create_tutorial_wizard()
        result = wizard.exec()
        if result:
            self._set_status("튜토리얼 완료")
        else:
            self._set_status("튜토리얼 닫힘")

    def _create_tutorial_wizard(self) -> object:
        """테스트에서 대체 가능한 튜토리얼 위저드 팩토리."""
        from naviertwin.gui.wizard.tutorial_wizard import TutorialWizard

        return TutorialWizard(self)

    def _show_doctor_report(self) -> None:
        """고객 지원용 런타임 진단 리포트를 GUI에서 표시한다."""
        from naviertwin.utils.doctor import build_doctor_report, format_doctor_report

        report = build_doctor_report(include_optional=True)
        status = str(report.get("status", "unknown"))
        self._set_status(f"환경 진단: {status}")
        QMessageBox.information(
            self,
            "NavierTwin 환경 진단",
            format_doctor_report(report),
        )

    def _create_support_bundle(self) -> None:
        outdir = QFileDialog.getExistingDirectory(
            self,
            "지원 번들 저장 폴더 선택",
            "",
        )
        if outdir:
            acceptance_json, acceptance_summary = (
                self._select_support_bundle_acceptance_artifacts()
            )
            self._create_support_bundle_path(
                Path(outdir),
                acceptance_json=acceptance_json,
                acceptance_summary=acceptance_summary,
            )

    def _create_support_bundle_path(
        self,
        outdir: Path,
        *,
        acceptance_json: Path | None = None,
        acceptance_summary: Path | None = None,
    ) -> None:
        """고객 지원용 진단 번들을 생성하고 결과를 표시한다."""
        from naviertwin.utils.support_bundle import build_support_bundle

        bundled_acceptance_json = acceptance_json or self._support_bundle_acceptance_json_path()
        bundled_acceptance_summary = (
            acceptance_summary or self._support_bundle_acceptance_summary_path()
        )
        try:
            metadata = build_support_bundle(
                outdir,
                preflight=self._support_bundle_preflight_path(),
                include_optional=True,
                zip_bundle=True,
                acceptance_json=bundled_acceptance_json,
                acceptance_summary=bundled_acceptance_summary,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status("지원 번들 생성 실패")
            QMessageBox.warning(self, "지원 번들 생성 실패", str(exc))
            return

        status = str(metadata.get("status", "unknown"))
        zip_name = str(metadata.get("zip_path", "support-bundle.zip"))
        zip_path = Path(zip_name)
        if not zip_path.is_absolute():
            zip_path = outdir / zip_path
        self._set_status(f"지원 번들 생성: {status}")
        QMessageBox.information(
            self,
            "지원 번들 생성 완료",
            f"상태: {status}\n저장 위치: {zip_path}",
        )

    def _select_support_bundle_acceptance_artifacts(self) -> tuple[Path | None, Path | None]:
        """최근 acceptance가 없을 때 지원 번들에 첨부할 리포트를 수동 선택한다."""
        if (
            self._support_bundle_acceptance_json_path() is not None
            or self._support_bundle_acceptance_summary_path() is not None
        ):
            return None, None
        reply = QMessageBox.question(
            self,
            "acceptance 리포트 첨부",
            "기존 acceptance JSON/Markdown 리포트를 지원 번들에 첨부하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return None, None

        json_path, _ = QFileDialog.getOpenFileName(
            self,
            "acceptance JSON 선택",
            "",
            "JSON (*.json)",
        )
        if not json_path:
            return None, None
        summary_path, _ = QFileDialog.getOpenFileName(
            self,
            "acceptance Markdown 선택(선택)",
            "",
            "Markdown (*.md);;All Files (*)",
        )
        return Path(json_path), Path(summary_path) if summary_path else None

    def _support_bundle_preflight_path(self) -> Path | None:
        """지원 번들에 포함할 현재 Import 탭 CFD 입력 경로를 반환한다."""
        path_text = self._import_panel._path_edit.text().strip()
        if not path_text:
            return None
        path = Path(path_text)
        return path if path.exists() else None

    def _support_bundle_acceptance_json_path(self) -> Path | None:
        """지원 번들에 포함할 최근 acceptance JSON 경로를 반환한다."""
        path = self._last_acceptance_json
        return path if path is not None and path.exists() else None

    def _support_bundle_acceptance_summary_path(self) -> Path | None:
        """지원 번들에 포함할 최근 acceptance Markdown 요약 경로를 반환한다."""
        path = self._last_acceptance_summary
        return path if path is not None and path.exists() else None

    def _inspect_support_bundle(self) -> None:
        """받은 고객 지원 번들 ZIP을 GUI에서 read-only로 점검한다."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "지원 번들 ZIP 선택",
            "",
            "ZIP (*.zip);;All Files (*)",
        )
        if not path:
            return
        self._inspect_support_bundle_path(Path(path))

    def _inspect_support_bundle_path(self, path: Path) -> None:
        """고객 지원 번들 점검 결과를 GUI에 표시한다."""
        from naviertwin.utils.support_bundle import (
            format_support_bundle_inspection,
            inspect_support_bundle,
        )

        try:
            report = inspect_support_bundle(path)
        except Exception as exc:  # noqa: BLE001
            self._set_status("지원 번들 점검 실패")
            QMessageBox.warning(self, "지원 번들 점검 실패", str(exc))
            return

        status = str(report.get("status", "unknown"))
        message = format_support_bundle_inspection(report)
        self._set_status(f"지원 번들 점검: {status}")
        if status == "ok":
            QMessageBox.information(self, "지원 번들 점검 완료", message)
        else:
            QMessageBox.warning(self, "지원 번들 점검 실패", message)

    def _open_recent_project(self) -> None:
        action: QAction = self.sender()  # type: ignore[assignment]
        path_text = action.data()
        if not isinstance(path_text, str):
            return
        path = Path(path_text)
        if not path.exists():
            self._remove_recent_project(path)
            QMessageBox.warning(
                self,
                "최근 프로젝트 열기 실패",
                f"파일을 찾을 수 없습니다:\n{path}",
            )
            return
        if not self._open_selected_path(path):
            self._remove_recent_project(path)

    def _check_for_updates(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "릴리스 메타데이터 선택",
            "",
            "Release Metadata (*.json);;All Files (*)",
        )
        if path:
            self._check_for_updates_path(Path(path))

    def _check_for_updates_path(self, path: Path) -> None:
        """선택한 릴리스 메타데이터로 업데이트 상태를 표시한다."""
        from naviertwin.utils.updater import check_for_update

        try:
            result = check_for_update(path)
        except (OSError, ValueError) as exc:
            self._set_status("업데이트 확인 실패")
            QMessageBox.warning(self, "업데이트 확인 실패", str(exc))
            return

        title, message = format_update_check_message(result)
        self._set_status(f"{title}: {result.latest_version}")
        if result.update_available:
            self._show_update_available_dialog(title, message, result)
        else:
            QMessageBox.information(self, title, message)

    def _show_update_available_dialog(
        self,
        title: str,
        message: str,
        result: UpdateCheckResult,
    ) -> None:
        """업데이트가 있을 때 다운로드/복사 handoff 버튼을 제공한다."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle(title)
        box.setText(message)
        open_button = box.addButton("다운로드 열기", QMessageBox.ButtonRole.AcceptRole)
        copy_button = box.addButton("URL 복사", QMessageBox.ButtonRole.ActionRole)
        verify_button = box.addButton("설치파일 검증", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()

        clicked = box.clickedButton()
        if clicked is open_button:
            self._open_update_download(result.url)
        elif clicked is copy_button:
            self._copy_update_url(result.url)
        elif clicked is verify_button:
            self._verify_update_artifact(result)

    def _open_update_download(self, url: str) -> bool:
        """검증된 릴리스 다운로드 URL을 기본 브라우저로 연다."""
        if not url:
            self._set_status("업데이트 다운로드 URL 없음")
            QMessageBox.warning(self, "업데이트 열기 실패", "다운로드 URL이 없습니다.")
            return False
        opened = QDesktopServices.openUrl(QUrl(url))
        if opened:
            self._set_status("업데이트 다운로드 URL을 열었습니다")
            return True
        self._copy_update_url(url)
        QMessageBox.warning(
            self,
            "업데이트 열기 실패",
            "브라우저를 열 수 없어 다운로드 URL을 클립보드에 복사했습니다.",
        )
        return False

    def _copy_update_url(self, url: str) -> bool:
        """검증된 릴리스 다운로드 URL을 클립보드에 복사한다."""
        if not url:
            self._set_status("업데이트 다운로드 URL 없음")
            QMessageBox.warning(self, "업데이트 복사 실패", "다운로드 URL이 없습니다.")
            return False
        QApplication.clipboard().setText(url)
        self._set_status("업데이트 다운로드 URL을 클립보드에 복사했습니다")
        return True

    def _verify_update_artifact(self, result: UpdateCheckResult) -> bool:
        """사용자가 선택한 설치 파일을 signed metadata SHA256으로 검증한다."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "업데이트 설치파일 선택",
            "",
            "NavierTwin Installer (NavierTwinSetup.exe);;Windows Executable (*.exe);;All Files (*)",
        )
        if not path:
            self._set_status("업데이트 설치파일 검증 취소")
            return False
        return self._verify_update_artifact_path(result, Path(path))

    def _verify_update_artifact_path(
        self,
        result: UpdateCheckResult,
        path: Path,
    ) -> bool:
        """선택된 설치 파일을 update-check 결과의 SHA256과 비교한다."""
        from naviertwin.utils.updater import verify_release_artifact

        try:
            verification = verify_release_artifact(path, expected_sha256=result.sha256)
        except (OSError, ValueError) as exc:
            self._set_status("업데이트 설치파일 검증 실패")
            QMessageBox.warning(self, "설치파일 검증 실패", str(exc))
            return False

        if verification.verified:
            self._set_status("업데이트 설치파일 검증 성공")
            QMessageBox.information(
                self,
                "설치파일 검증 성공",
                (
                    "다운로드한 설치 파일의 SHA256이 signed metadata와 일치합니다.\n\n"
                    f"파일: {verification.path}\n"
                    f"크기: {verification.size_bytes} bytes\n"
                    f"SHA256: {verification.actual_sha256}"
                ),
            )
            return True

        self._set_status("업데이트 설치파일 검증 실패")
        QMessageBox.warning(
            self,
            "설치파일 검증 실패",
            (
                "다운로드한 설치 파일의 SHA256이 signed metadata와 일치하지 않습니다.\n\n"
                f"파일: {verification.path}\n"
                f"기대값: {verification.expected_sha256}\n"
                f"실제값: {verification.actual_sha256}"
            ),
        )
        return False

    # ──────────────────────────────────────────────────────────────────
    # 헬퍼
    # ──────────────────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        self._status_label.setText(msg)

    def _load_gui_config(self) -> NavierTwinConfig:
        """GUI 설정을 로드하되 손상된 설정은 기본값으로 폴백한다."""
        try:
            return load_config(self._config_path)
        except Exception:  # noqa: BLE001
            return NavierTwinConfig()

    def _save_gui_config(self) -> None:
        """현재 GUI 설정을 저장한다."""
        try:
            save_config(self._config, self._config_path)
        except OSError:
            # 설정 저장 실패가 프로젝트 로드/저장 상태 메시지를 덮지 않게 한다.
            return

    def _remember_recent_project(self, path: Path) -> None:
        """최근 프로젝트 목록에 path를 최신 항목으로 저장한다."""
        path_text = str(path.expanduser().resolve())
        existing = [
            item for item in self._config.recent_projects
            if str(Path(item).expanduser().resolve()) != path_text
        ]
        self._config.recent_projects = [path_text, *existing][:10]
        self._save_gui_config()
        self._refresh_recent_projects_menu()

    def _remove_recent_project(self, path: Path) -> None:
        """존재하지 않는 최근 프로젝트를 목록에서 제거한다."""
        path_text = str(path.expanduser().resolve())
        self._config.recent_projects = [
            item for item in self._config.recent_projects
            if str(Path(item).expanduser().resolve()) != path_text
        ]
        self._save_gui_config()
        self._refresh_recent_projects_menu()

    def _refresh_recent_projects_menu(self) -> None:
        """File 메뉴의 최근 프로젝트 목록을 설정과 동기화한다."""
        menu = getattr(self, "_recent_projects_menu", None)
        if menu is None:
            return

        menu.clear()
        if not self._config.recent_projects:
            empty = QAction("최근 프로젝트 없음", self)
            empty.setEnabled(False)
            menu.addAction(empty)
            return

        for path_text in self._config.recent_projects:
            path = Path(path_text)
            action = QAction(path.name or path_text, self)
            action.setToolTip(path_text)
            action.setData(path_text)
            action.triggered.connect(self._open_recent_project)
            menu.addAction(action)

    def _current_project_path(self) -> Path | None:
        """ExportPanel 경로 입력에서 현재 .ntwin 프로젝트 경로를 추출한다."""
        path_text = self._export_panel._path_edit.text().strip()
        if not path_text:
            return None
        path = Path(path_text)
        return path if path.suffix.lower() == ".ntwin" else None

    def _extract_project_metadata(
        self, dataset: object, engine: object | None
    ) -> dict[str, object]:
        """복원용 프로젝트 메타데이터를 dataset 우선으로 추출한다."""
        dataset_meta = getattr(dataset, "metadata", None)
        if isinstance(dataset_meta, Mapping) and "project_metadata" in dataset_meta:
            project_meta = dataset_meta.get("project_metadata")
            normalized = self._normalize_project_metadata(project_meta)
            if normalized or isinstance(project_meta, Mapping):
                return normalized

        engine_meta = getattr(engine, "project_metadata", None) if engine is not None else None
        return self._normalize_project_metadata(engine_meta)

    def _normalize_project_metadata(self, metadata: object) -> dict[str, object]:
        """ModelPanel이 기대하는 dict[str, object] 형태로 정규화한다."""
        if not isinstance(metadata, Mapping):
            return {}

        normalized: dict[str, object] = {}
        for key, value in metadata.items():
            if isinstance(key, str):
                normalized[key] = value
        return normalized

    def _record_model_comparison(self, model_type: str, model: object) -> None:
        """학습 완료 모델의 검증 지표를 Compare 탭에 누적한다."""
        metrics = self._extract_validation_metrics(model)
        if metrics is None:
            return
        self._model_compare_results[
            self._format_compare_model_label(model_type, model)
        ] = metrics
        self.update_compare_dashboard(self._model_compare_results)

    def _extract_validation_metrics(self, model: object) -> dict[str, float] | None:
        """모델 메타데이터에서 CompareWidget이 사용할 RMSE/R²를 추출한다."""
        metadata = getattr(model, "training_metadata", None)
        if not isinstance(metadata, Mapping):
            return None
        raw_metrics = metadata.get("validation_metrics")
        if not isinstance(raw_metrics, Mapping):
            return None

        rmse = self._finite_float(raw_metrics.get("rmse"))
        r2 = self._finite_float(raw_metrics.get("r2"))
        if rmse is None or r2 is None:
            return None
        return {"rmse": rmse, "r2": r2}

    @staticmethod
    def _finite_float(value: object) -> float | None:
        """유한한 float 값만 CompareWidget 입력으로 허용한다."""
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _format_compare_model_label(model_type: str, model: object) -> str:
        """Compare 탭에 표시할 모델 이름을 정한다."""
        class_name = type(model).__name__
        if class_name and class_name != "object":
            return class_name
        return model_type.upper() if model_type else "Model"

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._confirm_on_close:
            self._stop_api_server_on_close()
            event.accept()
            return

        reply = QMessageBox.question(
            self,
            "종료 확인",
            "NavierTwin을 종료하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._stop_api_server_on_close()
            event.accept()
        else:
            event.ignore()

    def _stop_api_server_on_close(self) -> None:
        """윈도우 종료 시 GUI가 시작한 서버 프로세스를 정리한다."""
        process = self._server_process
        if process is None or process.state() == QProcess.ProcessState.NotRunning:
            return
        process.terminate()
        if not process.waitForFinished(1000):
            process.kill()
            process.waitForFinished(1000)
        self._server_process = None
