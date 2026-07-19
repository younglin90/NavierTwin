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

from PySide6.QtCore import QPoint, QProcess, QRect, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
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
from naviertwin.core.data_model import TwinWorkspace
from naviertwin.gui.panels.analyze_panel import AnalyzePanel, analysis_result_field
from naviertwin.gui.panels.explainability_panel import ExplainabilityPanel
from naviertwin.gui.panels.export_panel import ExportPanel
from naviertwin.gui.panels.import_panel import ImportPanel
from naviertwin.gui.panels.library_panel import LibraryPanel
from naviertwin.gui.panels.model_panel import ModelPanel
from naviertwin.gui.panels.reduce_panel import ReducePanel
from naviertwin.gui.panels.twin_panel import TwinPanel
from naviertwin.gui.widgets.pipeline_tabs import PipelineTabWidget
from naviertwin.utils.config import NavierTwinConfig, load_config, save_config
from naviertwin.utils.updater import UpdateCheckResult


def open_file_filter() -> str:
    """File → Open에서 사용할 프로젝트/CFD 통합 파일 필터."""
    from naviertwin.core.cfd_reader import ReaderFactory
    from naviertwin.gui.panels.import_panel import cfd_file_filter

    registered = " ".join(map(lambda ext: f"*{ext}", ReaderFactory.registered_extensions()))
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


def _paths_from_texts(paths: list[str]) -> list[Path]:
    """문자열 경로 목록을 Path 목록으로 바꾼다."""
    return list(map(Path, paths))


def _join_path_csv(paths: list[Path]) -> str:
    """Path 목록을 CLI CSV 인자로 만든다."""
    return ",".join(map(str, paths))


def _split_csv_preserve_empty(value: str) -> list[str]:
    """빈 칸을 보존하며 쉼표 구분 문자열을 정리한다."""
    return list(map(str.strip, value.split(",")))


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
        self.workspace = TwinWorkspace()
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

        # Feature Pack 누락 자동 감지 + 첫 실행 시 사용자에게 안내.
        # showEvent 직후 한 번만 동작하도록 QTimer.singleShot 사용.
        from PySide6.QtCore import QTimer  # noqa: PLC0415

        QTimer.singleShot(300, self._prompt_missing_feature_packs_once)
        self._install_combo_close_fix()

    # ──────────────────────────────────────────────────────────────────
    # 초기화
    # ──────────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        qss = _load_stylesheet(self._config.theme)
        app = QApplication.instance()
        if app is None:
            return
        # QApplication.setStyleSheet() re-polishes every live widget in the
        # whole application (not just this window), so it is O(total widget
        # count). Guard against redundant re-application with an unchanged
        # stylesheet — this is a no-op in normal single-window usage but
        # matters when many MainWindow instances are constructed in the same
        # process (e.g. GUI test suites), where each construction would
        # otherwise re-polish an ever-growing widget tree and slow down
        # quadratically.
        if app.styleSheet() == qss:
            return
        app.setStyleSheet(qss)

    def _install_combo_close_fix(self) -> None:
        """QSS 적용 후에도 QComboBox popup 이 클릭 선택 시 닫히도록 보정한다."""
        from naviertwin.gui.utils.combo_fix import (
            apply_to_widget_tree,
            install_combo_close_filter,
        )

        app = QApplication.instance()
        if app is not None:
            install_combo_close_filter(app)
        apply_to_widget_tree(self)

    def _setup_panels(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        self._tabs = PipelineTabWidget(max_columns=3)
        self._tabs.setDocumentMode(True)
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)
        self._tabs.setUsesScrollButtons(False)
        self._tabs.setElideMode(Qt.TextElideMode.ElideRight)

        from naviertwin.utils.i18n import Translator

        self._t = Translator(lang=self._config.language)
        self.setWindowTitle(self._t("app.title", "NavierTwin — CFD Digital Twin"))

        self._import_panel = ImportPanel()
        self._analyze_panel = AnalyzePanel(use_embedded_viewer=False)
        self._reduce_panel = ReducePanel()
        self._model_panel = ModelPanel()
        self._twin_panel = TwinPanel()
        self._export_panel = ExportPanel()
        self._explain_panel: ExplainabilityPanel | None = ExplainabilityPanel()
        self._library_panel = LibraryPanel()

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

        self._analyze_workbench = self._build_analyze_workbench()
        self._model_workbench = self._build_model_workbench()

        self._tab_title_specs: list[tuple[QWidget, str, str, str]] = [
            (self._import_panel, "panel.import", "①", "Import"),
            (self._analyze_workbench, "panel.analyze", "②", "Analyze"),
            (self._reduce_panel, "panel.reduce", "③", "Reduce"),
            (self._model_workbench, "panel.model", "④", "Model"),
            (self._twin_panel, "panel.twin", "⑤", "Twin"),
            (self._export_panel, "panel.export", "⑥", "Export"),
        ]

        tab_index = 0
        while tab_index < len(self._tab_title_specs):
            widget, key, num, default = self._tab_title_specs[tab_index]
            tab_index += 1
            self._tabs.addTab(widget, self._localized_tab_title(key, num, default))

        vbox.addWidget(self._tabs)
        self._setup_persistent_viewer()
        self._setup_library_search_dialog()

    def _build_analyze_workbench(self) -> QWidget:
        """Analyze 탭 안에 분석 화면과 후처리 도구를 함께 배치한다."""
        if self._postproc_panel is None:
            return self._analyze_panel

        workbench = QTabWidget()
        workbench.setDocumentMode(True)
        workbench.setObjectName("analyzeWorkbenchTabs")
        workbench.addTab(self._analyze_panel, self._t("panel.analyze", "Analyze"))
        workbench.addTab(self._postproc_panel, self._t("panel.post_tools", "Post-Tools"))
        return workbench

    def _build_model_workbench(self) -> QWidget:
        """Model 탭 안에 모델링, 비교, 설명 화면을 함께 배치한다."""
        if self._compare_panel is None and self._explain_panel is None:
            return self._model_panel

        workbench = QTabWidget()
        workbench.setDocumentMode(True)
        workbench.setObjectName("modelWorkbenchTabs")
        workbench.addTab(self._model_panel, self._t("panel.model", "Model"))
        if self._compare_panel is not None:
            workbench.addTab(self._compare_panel, self._t("panel.compare", "Compare"))
        if self._explain_panel is not None:
            workbench.addTab(self._explain_panel, self._t("panel.explain", "Explain"))
        return workbench

    def _refresh_analyze_workbench_tabs(self) -> None:
        """Analyze 내부 하위 탭도 현재 언어에 맞춰 갱신한다."""
        if not isinstance(self._analyze_workbench, QTabWidget):
            return

        analyze_idx = self._analyze_workbench.indexOf(self._analyze_panel)
        if analyze_idx >= 0:
            self._analyze_workbench.setTabText(
                analyze_idx,
                self._t("panel.analyze", "Analyze"),
            )
        if self._postproc_panel is not None:
            post_idx = self._analyze_workbench.indexOf(self._postproc_panel)
            if post_idx >= 0:
                self._analyze_workbench.setTabText(
                    post_idx,
                    self._t("panel.post_tools", "Post-Tools"),
                )

    def _refresh_model_workbench_tabs(self) -> None:
        """Model 내부 하위 탭도 현재 언어에 맞춰 갱신한다."""
        if not isinstance(self._model_workbench, QTabWidget):
            return

        model_idx = self._model_workbench.indexOf(self._model_panel)
        if model_idx >= 0:
            self._model_workbench.setTabText(
                model_idx,
                self._t("panel.model", "Model"),
            )
        if self._compare_panel is not None:
            compare_idx = self._model_workbench.indexOf(self._compare_panel)
            if compare_idx >= 0:
                self._model_workbench.setTabText(
                    compare_idx,
                    self._t("panel.compare", "Compare"),
                )
        if self._explain_panel is not None:
            explain_idx = self._model_workbench.indexOf(self._explain_panel)
            if explain_idx >= 0:
                self._model_workbench.setTabText(
                    explain_idx,
                    self._t("panel.explain", "Explain"),
                )

    def _setup_persistent_viewer(self) -> None:
        """탭과 독립적으로 유지되는 전역 3D viewer dock을 구성한다."""
        from naviertwin.gui.widgets.vtk_viewer import VtkViewer

        self._global_viewer = VtkViewer()
        self._viewer_dock = QDockWidget("3D Viewer", self)
        self._viewer_dock.setObjectName("global3DViewerDock")
        self._viewer_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self._viewer_dock.setWidget(self._global_viewer)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._viewer_dock)
        self.resizeDocks([self._viewer_dock], [340], Qt.Orientation.Horizontal)

    def _setup_library_search_dialog(self) -> None:
        """Library/기능 검색을 도움말에서 여는 독립 팝업 창으로 구성한다."""
        self._library_dialog = QDialog(self)
        self._library_dialog.setObjectName("librarySearchDialog")
        self._library_dialog.setWindowTitle(self._t("panel.library", "Library"))
        self._library_dialog.setModal(False)
        self._library_dialog.setMinimumSize(980, 680)
        layout = QVBoxLayout(self._library_dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._library_panel)

    def _localized_tab_title(self, key: str, num: str, default: str) -> str:
        """번호 prefix가 붙은 탭 제목을 현재 언어로 구성한다."""
        return f"{num} {self._t(key, default)}"

    def set_language(self, lang: str) -> None:
        """런타임 언어 전환 (탭 제목과 보기 메뉴 갱신)."""
        self._t.set_language(lang)
        self._config.language = lang  # type: ignore[assignment]
        self.setWindowTitle(self._t("app.title", "NavierTwin — CFD Digital Twin"))
        tab_index = 0
        while tab_index < len(self._tab_title_specs):
            widget, key, num, default = self._tab_title_specs[tab_index]
            tab_index += 1
            index = self._tabs.indexOf(widget)
            if index >= 0:
                self._tabs.setTabText(index, self._localized_tab_title(key, num, default))
        self._refresh_analyze_workbench_tabs()
        self._refresh_model_workbench_tabs()
        library_dialog = getattr(self, "_library_dialog", None)
        if library_dialog is not None:
            library_dialog.setWindowTitle(self._t("panel.library", "Library"))
        # Post-Tools 패널 retranslate (있으면)
        if self._postproc_panel is not None:
            try:
                self._postproc_panel.set_language(lang)
            except Exception:  # noqa: BLE001
                pass
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
        self._model_compare_results = dict(
            map(lambda item: (item[0], dict(item[1])), results.items())
        )
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
        self._view_menu = mb.addMenu(self._t("menu.view", "View(&V)"))
        self._refresh_view_menu()

        # 도구 메뉴
        self._tools_menu = mb.addMenu("도구(&T)")
        benchmark_action = QAction("벤치마크 실행(&B)", self)
        benchmark_action.triggered.connect(self._run_benchmark)
        self._tools_menu.addAction(benchmark_action)

        pipeline_demo_action = QAction("파이프라인 데모 실행(&P)", self)
        pipeline_demo_action.triggered.connect(self._run_pipeline_demo)
        self._tools_menu.addAction(pipeline_demo_action)

        # 데모 데이터 (v5.2) — 웹 GUI 데모 카탈로그의 시계열 데모를 데스크톱에서도.
        # 파일 없이 즉시 로드돼 전체 파이프라인을 바로 체험할 수 있다.
        demo_menu = self._tools_menu.addMenu("데모 데이터 로드(&D)")
        for kind, label in (
            ("filament", "소용돌이 필라멘트 (불연속 — 어려운 기준)"),
            ("advecting", "이류 Taylor–Green (부드러움 — ROM 적합)"),
            ("waves", "진행파 2모드 (DMD 적합)"),
            ("karman", "카르만 와열 (실제 LBM 해석 · 진짜 구멍)"),
        ):
            action = QAction(label, self)
            action.triggered.connect(
                lambda _checked=False, k=kind: self._load_demo_dataset(k)
            )
            demo_menu.addAction(action)

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
        library_action = QAction("기능 검색(&L)", self)
        library_action.setShortcut("Ctrl+K")
        library_action.triggered.connect(self._show_library_search)
        self._help_menu.addAction(library_action)

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

        view_menu.setTitle(self._t("menu.view", "View(&V)"))
        view_menu.clear()
        tab_index = 0
        while tab_index < tabs.count():
            title = tabs.tabText(tab_index)
            action = QAction(
                self._t("view.tab_action", "{title} Tab").format(title=title),
                self,
            )
            if tab_index < 9:
                action.setShortcut(f"Ctrl+{tab_index + 1}")
            action.setData(tab_index)
            action.triggered.connect(self._switch_tab)
            view_menu.addAction(action)
            tab_index += 1

        viewer_dock = getattr(self, "_viewer_dock", None)
        if viewer_dock is not None:
            view_menu.addSeparator()
            view_menu.addAction(viewer_dock.toggleViewAction())

        if getattr(self, "_library_dialog", None) is not None:
            library_action = QAction(self._t("panel.library", "Library"), self)
            library_action.triggered.connect(self._show_library_search)
            view_menu.addAction(library_action)

        view_menu.addSeparator()
        theme_specs = (
            ("dark", "view.theme.dark", "Dark Theme"),
            ("light", "view.theme.light", "Light Theme"),
        )
        theme_index = 0
        while theme_index < len(theme_specs):
            theme, key, default = theme_specs[theme_index]
            theme_index += 1
            label = self._t(key, default)
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(self._config.theme == theme)
            action.setData(theme)
            action.triggered.connect(self._switch_theme)
            view_menu.addAction(action)

        view_menu.addSeparator()
        language_specs = (
            ("ko", "view.language.ko", "Korean"),
            ("en", "view.language.en", "English"),
        )
        language_index = 0
        while language_index < len(language_specs):
            lang, key, default = language_specs[language_index]
            language_index += 1
            label = self._t(key, default)
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
        self._analyze_panel.analysis_done.connect(self._on_analysis_done)

        # Export 완료 상태 업데이트
        self._export_panel.export_done.connect(self._on_export_done)
        self._export_panel.project_loaded.connect(self._on_project_loaded)

        # Twin 예측 완료
        self._twin_panel.prediction_done.connect(self._on_twin_prediction_done)
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
        self._twin_panel.uq_done.connect(
            lambda result: self._set_status(
                f"UQ 완료: {result.get('method', 'UQ')}, "
                f"N={result.get('n_samples', '?')}"
            )
        )
        self._twin_panel.applied_done.connect(
            lambda result: self._set_status(
                f"Applied 계산 완료: {result.get('calculator', 'calculator')}"
            )
        )
        if self._explain_panel is not None:
            self._explain_panel.explanation_done.connect(
                lambda result: self._set_status(
                    f"SHAP 설명 완료: {len(result.get('feature_names', []))} features"
                )
            )

        self._library_panel.capability_done.connect(
            lambda cap_id, _: self._set_status(f"기능 데모 완료: {cap_id}")
        )
        self._library_panel.navigate_requested.connect(self._on_library_navigate)

        # Simulation 결과 → 상태바 + 전역 viewer 연동 훅
        if self._simulation_panel is not None:
            self._simulation_panel.simulation_done.connect(self._on_simulation_done)

    # ------------------------------------------------------------------
    def _on_simulation_done(self, kind: str, result: object) -> None:
        """SimulationPanel 결과를 viewer 로 전달 (가능한 경우)."""
        self._set_status(
            f"시뮬레이션 완료: {kind} — {getattr(result, 'get', lambda *_: '')('summary', '')}"
        )
        # viewer 가 있으면 시뮬레이션 결과를 표시 가능한 형태로 전달한다.
        viewer = getattr(self, "_global_viewer", None)
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

    def _on_analysis_done(self, method: str, _: object) -> None:
        """분석 완료 후 전역 viewer를 분석 결과 field로 동기화한다."""
        self._set_status(f"분석 완료 ({method})")
        field_name = analysis_result_field(method)
        if field_name:
            self._show_field_in_global_viewer(field_name)

    def _on_twin_prediction_done(self, prediction: object) -> None:
        """Twin 예측 field를 현재 CFD mesh에 붙이고 3D viewer에 표시한다."""
        self._set_status("예측 완료")
        dataset = self._latest_dataset
        if dataset is None:
            return

        try:
            import numpy as np

            mesh = dataset.mesh  # type: ignore[union-attr]
            values = np.asarray(prediction, dtype=float).reshape(-1)
            n_points = int(getattr(dataset, "n_points", 0))
            n_cells = int(getattr(dataset, "n_cells", 0))
            if self._attach_structured_twin_prediction(values, mesh, dataset):
                return
            field_name = self._prediction_field_name()
            if values.size == n_cells:
                mesh.cell_data[field_name] = values
            elif values.size == n_points:
                mesh.point_data[field_name] = values
            else:
                self._set_status(
                    f"예측 완료 — viewer 표시 생략(shape={values.shape}, "
                    f"points={n_points}, cells={n_cells})"
                )
                return

            if field_name not in dataset.field_names:  # type: ignore[operator]
                dataset.field_names.append(field_name)  # type: ignore[union-attr]
            self._analyze_panel._sync_result_field_to_viewer(field_name)
            self._show_field_in_global_viewer(field_name)
            self._set_status(f"예측 완료 — 3D viewer 표시: {field_name}")
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"예측 완료 — viewer 표시 실패: {exc}")

    def _attach_structured_twin_prediction(
        self,
        values: object,
        mesh: object,
        dataset: object,
    ) -> bool:
        """Attach multi-output PhysicsNeMo CFD predictions to the current mesh."""
        import numpy as np

        metadata = self._latest_prediction_metadata()
        output_fields = metadata.get("output_fields") if isinstance(metadata, Mapping) else None
        if not isinstance(output_fields, list) or not output_fields:
            return False

        array = np.asarray(values, dtype=float).reshape(-1)
        shown_field = ""
        attached = 0
        spec_index = 0
        while spec_index < len(output_fields):
            spec = output_fields[spec_index]
            spec_index += 1
            if not isinstance(spec, Mapping):
                continue
            try:
                start = int(spec.get("start", -1))
                end = int(spec.get("end", -1))
            except (TypeError, ValueError):
                continue
            if start < 0 or end <= start or end > array.size:
                continue
            raw_name = str(spec.get("display_name") or spec.get("field_name") or "field")
            field_name = self._safe_prediction_field_name(raw_name)
            part = array[start:end]
            location = str(spec.get("location", "point"))
            if location == "cell":
                mesh.cell_data[field_name] = part  # type: ignore[attr-defined]
            else:
                mesh.point_data[field_name] = part  # type: ignore[attr-defined]
            if field_name not in dataset.field_names:  # type: ignore[operator]
                dataset.field_names.append(field_name)  # type: ignore[union-attr]
            shown_field = shown_field or field_name
            attached += 1

        if attached == 0:
            return False
        if shown_field:
            self._analyze_panel._sync_result_field_to_viewer(shown_field)
            self._show_field_in_global_viewer(shown_field)
        self._set_status(f"예측 완료 — {attached}개 field를 3D viewer에 연결")
        return True

    def _latest_prediction_metadata(self) -> Mapping[str, object]:
        """Return metadata from the latest model/engine that produced predictions."""
        sources = (self._latest_engine, self._latest_surrogate, self._latest_reducer)
        source_index = 0
        while source_index < len(sources):
            source = sources[source_index]
            source_index += 1
            meta = getattr(source, "training_metadata", None)
            if isinstance(meta, Mapping):
                return meta
        return {}

    @staticmethod
    def _safe_prediction_field_name(raw_name: str) -> str:
        safe = "".join(map(lambda ch: ch if ch.isalnum() or ch == "_" else "_", raw_name))
        return f"twin_pred_{safe or 'field'}"

    def _prediction_field_name(self) -> str:
        """Reducer metadata에서 예측 대상 field 이름을 추출한다."""
        reducer = self._latest_reducer
        field = ""
        sources = (reducer, self._latest_surrogate, self._latest_engine)
        source_index = 0
        while source_index < len(sources):
            source = sources[source_index]
            source_index += 1
            meta = getattr(source, "training_metadata", None)
            if isinstance(meta, Mapping):
                candidate = meta.get("field_name")
                if isinstance(candidate, str):
                    field = candidate
                    break
        if not field and self._latest_dataset is not None:
            fields = getattr(self._latest_dataset, "field_names", [])
            if fields:
                field = str(fields[0])
        if field == "U":
            field = "U_mag"
        return self._safe_prediction_field_name(field)

    def _on_dataset_loaded(self, dataset: object) -> None:
        self._latest_dataset = dataset
        self._latest_reducer = None
        self._latest_surrogate = None
        self._latest_operator = None
        self._latest_engine = None
        metadata = getattr(dataset, "metadata", {}) or {}
        source = str(metadata.get("source") or metadata.get("source_file") or "")
        try:
            self.workspace.load_single_dataset(
                dataset,
                name="NavierTwin desktop project",
                source=source,
            )
        except Exception:  # noqa: BLE001 - legacy/third-party dataset duck types
            # Canonical state enriches the legacy Qt path; it must not block
            # existing panel consumers that accept lighter dataset protocols.
            pass
        self._set_status(
            f"데이터셋 로드 완료 — {dataset.n_points} pts, "  # type: ignore[union-attr]
            f"{dataset.n_cells} cells, {dataset.n_time_steps} steps"
        )
        self._analyze_panel.set_dataset(dataset)  # type: ignore[arg-type]
        self._load_global_viewer_dataset(dataset)
        self._reduce_panel.set_dataset(dataset)    # type: ignore[arg-type]
        self._model_panel.set_dataset(dataset)     # type: ignore[arg-type]
        self._export_panel.set_dataset(dataset)    # type: ignore[arg-type]
        self._export_panel.set_canonical_project(self.workspace.project)
        if self._explain_panel is not None:
            self._explain_panel.set_dataset(dataset)
        if self._postproc_panel is not None:
            self._postproc_panel.set_dataset(dataset)
        self._library_panel.set_dataset(dataset)
        # Import 탭 완료 후 Analyze 탭으로 자동 이동
        self._tabs.setCurrentIndex(1)

    def _load_global_viewer_dataset(self, dataset: object) -> None:
        """전역 viewer에 CFD dataset을 로드한다."""
        viewer = getattr(self, "_global_viewer", None)
        if viewer is None:
            return
        try:
            viewer.load_dataset(dataset)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"전역 3D viewer 로드 실패: {exc}")

    def _show_field_in_global_viewer(self, field_name: str) -> None:
        """전역 viewer field 목록을 갱신하고 지정 field를 표시한다."""
        viewer = getattr(self, "_global_viewer", None)
        if viewer is None:
            return
        try:
            viewer.refresh_fields(prefer_field=field_name)
            viewer.show_field(field_name)
            dock = getattr(self, "_viewer_dock", None)
            if dock is not None:
                dock.show()
                dock.raise_()
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"전역 3D viewer 표시 실패: {exc}")

    def _switch_to_model_workbench(self, subtab: QWidget | None = None) -> None:
        """Model 최상위 탭으로 이동하고 필요한 경우 내부 하위 탭도 선택한다."""
        self._tabs.setCurrentWidget(self._model_workbench)
        if subtab is not None and isinstance(self._model_workbench, QTabWidget):
            self._model_workbench.setCurrentWidget(subtab)

    def _switch_to_analyze_workbench(self, subtab: QWidget | None = None) -> None:
        """Analyze 최상위 탭으로 이동하고 필요한 경우 내부 하위 탭도 선택한다."""
        self._tabs.setCurrentWidget(self._analyze_workbench)
        if subtab is not None and isinstance(self._analyze_workbench, QTabWidget):
            self._analyze_workbench.setCurrentWidget(subtab)

    def _show_library_search(self) -> None:
        """기능 검색 팝업을 화면 중앙에 적절한 크기로 연다."""
        dialog = getattr(self, "_library_dialog", None)
        if dialog is None:
            return
        self._position_library_search_dialog()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self._set_status("기능 검색 열림")

    def _position_library_search_dialog(self) -> None:
        """기능 검색 dialog를 현재 창/화면 중앙에 배치한다."""
        dialog = self._library_dialog
        screen = self.screen() or QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else QRect(0, 0, 1280, 800)
        width = min(max(int(self.width() * 0.78), 980), max(640, available.width() - 80))
        height = min(max(int(self.height() * 0.78), 680), max(520, available.height() - 80))
        dialog.resize(width, height)

        center = self.frameGeometry().center() if self.isVisible() else available.center()
        left = max(available.left(), min(center.x() - width // 2, available.right() - width + 1))
        top = max(available.top(), min(center.y() - height // 2, available.bottom() - height + 1))
        dialog.move(left, top)

    def _prompt_missing_feature_packs_once(self) -> None:
        """첫 실행 시 누락된 Feature Pack 을 검사하고 GUI Library 탭으로 안내.

        테스트/offscreen 환경, 또는 ``NAVIERTWIN_SKIP_FEATURE_PACK_PROMPT=1`` 이면
        팝업을 띄우지 않는다.
        """
        import os  # noqa: PLC0415

        if getattr(self, "_feature_pack_prompt_shown", False):
            return
        self._feature_pack_prompt_shown = True
        if os.environ.get("NAVIERTWIN_SKIP_FEATURE_PACK_PROMPT") == "1":
            return
        if os.environ.get("QT_QPA_PLATFORM", "").lower() in {"offscreen", "minimal"}:
            return
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return
        try:
            from naviertwin.utils.feature_packs import (  # noqa: PLC0415
                FEATURE_PACKS,
                feature_pack_status,
            )
        except Exception:
            return
        missing: list[str] = []
        pack_ids = list(FEATURE_PACKS)
        pack_index = 0
        while pack_index < len(pack_ids):
            pack_id = pack_ids[pack_index]
            pack_index += 1
            try:
                st = feature_pack_status(pack_id)
            except Exception:
                continue
            if not st.get("installed") or st.get("missing_modules"):
                missing.append(pack_id)
        if not missing:
            self._set_status("Feature Pack 상태: 모두 설치됨")
            return
        # 한 번만 알림. 사용자가 "Library 로 이동" 누르면 패널로 점프.
        from PySide6.QtWidgets import QMessageBox  # noqa: PLC0415

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Feature Pack 안내")
        missing_text = "\n  • ".join(missing)
        msg.setText(
            "다음 선택 기능 (Feature Pack) 이 아직 설치되지 않았습니다:\n\n"
            f"  • {missing_text}\n\n"
            "Library 탭에서 한 번의 클릭으로 PyPI 에서 직접 설치할 수 있습니다."
        )
        go_btn = msg.addButton("Library 탭으로 이동", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton("나중에", QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        if msg.clickedButton() is go_btn:
            try:
                idx = self._tabs.indexOf(self._library_panel)
                if idx >= 0:
                    self._tabs.setCurrentIndex(idx)
            except Exception:
                pass

    def _on_library_navigate(self, route: str) -> None:
        """Library 탭에서 요청한 기존 기능 탭으로 이동한다."""
        route_key = route.lower()
        if route_key.startswith("compare") and self._compare_panel is not None:
            self._switch_to_model_workbench(self._compare_panel)
            self._set_status(f"기능 탭 이동: {route}")
            return
        if route_key.startswith("explain") and self._explain_panel is not None:
            self._switch_to_model_workbench(self._explain_panel)
            self._set_status(f"기능 탭 이동: {route}")
            return
        if route_key.startswith("post-tools") and self._postproc_panel is not None:
            self._switch_to_analyze_workbench(self._postproc_panel)
            self._set_status(f"기능 탭 이동: {route}")
            return
        if route_key.startswith("library"):
            self._show_library_search()
            return
        if route_key.startswith("simulation"):
            self._set_status("Simulation은 workflow 최상위 탭에서 제거되었습니다.")
            return

        targets: list[tuple[str, QWidget | None]] = [
            ("import", self._import_panel),
            ("analyze", self._analyze_workbench),
            ("reduce", self._reduce_panel),
            ("model", self._model_workbench),
            ("twin", self._twin_panel),
            ("export", self._export_panel),
        ]
        target_index = 0
        while target_index < len(targets):
            key, widget = targets[target_index]
            target_index += 1
            if widget is not None and route_key.startswith(key):
                self._tabs.setCurrentWidget(widget)
                self._set_status(f"기능 탭 이동: {route}")
                return
        self._set_status(f"알 수 없는 기능 탭: {route}")

    def _on_reduction_done(self, method: str, reducer: object) -> None:
        self._latest_reducer = reducer
        self._set_status(f"차원 축소 완료 ({method})")
        self._model_panel.set_reducer(reducer)
        artifact = self._reduce_panel.get_reduction_artifact()
        if artifact is not None:
            self._model_panel.set_reduction_artifact(artifact)
        self._tabs.setCurrentIndex(3)

    def _on_model_trained(self, model_type: str, surrogate: object) -> None:
        if self._is_physics_ai_model(model_type, surrogate):
            try:
                engine = self._build_physics_ai_engine(model_type, surrogate)
                self._latest_surrogate = surrogate
                self._latest_engine = engine
                self.workspace.set_engine(engine)
                self._twin_panel.set_engine(engine)
                self._export_panel.set_engine(engine)
                if self._explain_panel is not None:
                    self._explain_panel.set_model(surrogate)
                self._set_status(
                    f"Physics AI 모델 학습 완료 ({model_type}) — Twin 직접 연결 완료"
                )
            except Exception as exc:
                self._set_status(
                    f"Physics AI 모델 학습 완료 ({model_type}) — Twin 연결 실패: {exc}"
                )
            self._record_model_comparison(model_type, surrogate)
            self._tabs.setCurrentWidget(self._twin_panel)
            return

        if self._is_direct_field_model(surrogate):
            try:
                engine = self._build_physics_ai_engine(model_type, surrogate)
                self._latest_surrogate = surrogate
                self._latest_engine = engine
                self.workspace.set_engine(engine)
                self._twin_panel.set_engine(engine)
                self._export_panel.set_engine(engine)
                if self._explain_panel is not None:
                    self._explain_panel.set_model(surrogate)
                self._set_status(
                    f"직접 CFD field surrogate 학습 완료 ({model_type}) — Twin 직접 연결 완료"
                )
            except Exception as exc:
                self._set_status(
                    f"직접 CFD field surrogate 학습 완료 ({model_type}) — Twin 연결 실패: {exc}"
                )
            self._record_model_comparison(model_type, surrogate)
            self._tabs.setCurrentWidget(self._twin_panel)
            return

        if self._is_operator_model(model_type, surrogate):
            self._latest_operator = surrogate
            self._export_panel.set_model(surrogate)
            self._set_status(
                f"연산자 학습 완료 ({model_type}) — TwinEngine 자동 연결 생략"
            )
            self._record_model_comparison(model_type, surrogate)
            self._switch_to_model_workbench(self._model_panel)
            return

        self._latest_surrogate = surrogate
        if self._explain_panel is not None:
            self._explain_panel.set_model(surrogate)
        if self._latest_reducer is not None:
            try:
                engine = self._build_engine(self._latest_reducer, surrogate)
                self._latest_engine = engine
                self.workspace.set_engine(engine)
                self._twin_panel.set_engine(engine)
                self._export_panel.set_engine(engine)
                self._set_status(f"모델 학습 완료 ({model_type}) — TwinEngine 연결 완료")
            except Exception as exc:
                self._set_status(f"모델 학습 완료 ({model_type}) — 엔진 구성 실패: {exc}")
        else:
            self._set_status(f"모델 학습 완료 ({model_type})")
        self._record_model_comparison(model_type, surrogate)
        self._tabs.setCurrentIndex(4)

    def _load_demo_dataset(self, kind: str) -> None:
        """데모 데이터셋을 메모리에서 생성해 전체 워크플로우에 로드한다 (v5.2).

        ``naviertwin.web.service`` 는 Qt/trame 비의존 워크플로우 계층이라
        데스크톱에서 그대로 재사용한다. karman 은 저장소 번들 데이터에서 즉시
        로드된다 (재계산 없음).
        """
        try:
            from naviertwin.web import service as web_service

            dataset = web_service.make_demo_dataset(kind=kind)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "데모 로드 실패", str(exc))
            return
        self._on_dataset_loaded(dataset)
        self._set_status(f"데모 데이터 로드 완료 — {kind}")

    def _on_project_loaded(self, dataset: object, engine: object | None) -> None:
        """Export 패널에서 로드한 프로젝트를 전체 워크플로우 상태로 복원한다."""
        self._on_dataset_loaded(dataset)
        load_warning = self._export_panel.last_project_load_warning()
        project_path = self._current_project_path()
        manifest_path = (
            project_path.with_suffix(".manifest.json")
            if project_path is not None
            else None
        )
        if manifest_path is not None and manifest_path.exists():
            try:
                from naviertwin.core.data_model import load_project_manifest

                canonical_project = load_project_manifest(manifest_path)
                runtime_complete = (
                    len(canonical_project.case_sets) == 1
                    and len(canonical_project.case_sets[0].cases) == 1
                )
                self.workspace.adopt_project(
                    canonical_project,
                    view_dataset=dataset,
                    engine=engine,
                    runtime_complete=runtime_complete,
                )
                self._export_panel.set_canonical_project(canonical_project)
            except Exception as exc:  # noqa: BLE001
                load_warning = f"canonical manifest 로드 실패: {exc}"
        if engine is not None:
            self._latest_engine = engine
            self.workspace.set_engine(engine)
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
            self.workspace.set_engine(None)
            self._set_status(f"프로젝트 부분 로드 완료 (dataset) — {load_warning}")
        else:
            self._latest_engine = None
            self.workspace.set_engine(None)
            self._set_status("프로젝트 로드 완료 (dataset)")
        loaded_meta = self._extract_project_metadata(dataset, engine)
        self._model_panel.set_loaded_metadata(loaded_meta)
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

    def _build_physics_ai_engine(self, model_type: str, model: object) -> object:
        """PhysicsNeMo/PINN 직접 예측 모델을 Twin 패널용 엔진으로 감싼다."""
        from naviertwin.core.digital_twin.physics_ai_engine import PhysicsAITwinEngine

        return PhysicsAITwinEngine.from_fitted_model(model, model_type=model_type)

    @staticmethod
    def _is_physics_ai_model(model_type: str, model: object) -> bool:
        """PhysicsNeMo/PINN 계열 모델인지 판정한다."""
        lower = model_type.lower()
        if lower.startswith(("physics", "physnemo", "physicsnemo")):
            return True
        module = type(model).__module__.lower()
        return ".physnemo" in module

    @staticmethod
    def _is_direct_field_model(model: object) -> bool:
        """Return True when models already predict full CFD fields."""
        meta = getattr(model, "training_metadata", None)
        if not isinstance(meta, Mapping):
            return False
        if bool(meta.get("direct_field_model")):
            return True
        output_fields = meta.get("output_fields")
        return isinstance(output_fields, list) and bool(output_fields)

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
                _paths_from_texts(paths),
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
            csv_snapshots=_join_path_csv(csv_paths),
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
            _paths_from_texts(paths),
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
            _paths_from_texts(paths),
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
            csv_snapshots=_join_path_csv(csv_paths),
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
            csv_snapshots=_join_path_csv(csv_paths),
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
        parts = _split_csv_preserve_empty(stripped)
        if len(parts) > 3:
            raise ValueError("검증 기준은 max_rmse,min_r2,max_relative_l2 순서로 최대 3개입니다.")
        parsed: list[float | None] = []
        part_index = 0
        while part_index < len(parts):
            part = parts[part_index]
            part_index += 1
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
        parts = _split_csv_preserve_empty(stripped)
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
        parts = _split_csv_preserve_empty(stripped)
        if len(parts) > 2:
            raise ValueError("SLO는 max_p95_ms,min_throughput_hz 순서로 최대 2개입니다.")

        parsed: list[float | None] = []
        part_index = 0
        while part_index < len(parts):
            part = parts[part_index]
            part_index += 1
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
        self.workspace.set_engine(engine)
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
            verification = verify_release_artifact(
                path,
                expected_sha256=result.sha256,
                installer_signing=result.installer_signing,
            )
        except (OSError, ValueError) as exc:
            self._set_status("업데이트 설치파일 검증 실패")
            QMessageBox.warning(self, "설치파일 검증 실패", str(exc))
            return False

        if verification.verified:
            authenticode = verification.authenticode or {}
            authenticode_line = (
                f"\nAuthenticode: {authenticode.get('status')}"
                if authenticode
                else ""
            )
            self._set_status("업데이트 설치파일 검증 성공")
            QMessageBox.information(
                self,
                "설치파일 검증 성공",
                (
                    "다운로드한 설치 파일의 SHA256이 signed metadata와 일치합니다.\n\n"
                    f"파일: {verification.path}\n"
                    f"크기: {verification.size_bytes} bytes\n"
                    f"SHA256: {verification.actual_sha256}"
                    f"{authenticode_line}"
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
        existing = list(
            filter(
                lambda item: str(Path(item).expanduser().resolve()) != path_text,
                self._config.recent_projects,
            )
        )
        self._config.recent_projects = [path_text, *existing][:10]
        self._save_gui_config()
        self._refresh_recent_projects_menu()

    def _remove_recent_project(self, path: Path) -> None:
        """존재하지 않는 최근 프로젝트를 목록에서 제거한다."""
        path_text = str(path.expanduser().resolve())
        self._config.recent_projects = list(
            filter(
                lambda item: str(Path(item).expanduser().resolve()) != path_text,
                self._config.recent_projects,
            )
        )
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

        project_index = 0
        while project_index < len(self._config.recent_projects):
            path_text = self._config.recent_projects[project_index]
            project_index += 1
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
        meta_items = list(metadata.items())
        meta_index = 0
        while meta_index < len(meta_items):
            key, value = meta_items[meta_index]
            meta_index += 1
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

        if self._ask_close_confirmation():
            self._stop_api_server_on_close()
            event.accept()
        else:
            event.ignore()

    def _ask_close_confirmation(self) -> bool:
        """종료 확인 다이얼로그를 메인 윈도우 중앙에 띄우고 결과를 반환한다."""
        dialog = self._build_close_confirmation_dialog()
        QTimer.singleShot(0, lambda: self._center_child_window(dialog))
        return dialog.exec() == QMessageBox.StandardButton.Yes

    def _build_close_confirmation_dialog(self) -> QMessageBox:
        """테스트 가능한 종료 확인 다이얼로그를 구성한다."""
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setWindowTitle("종료 확인")
        dialog.setText("NavierTwin을 종료하시겠습니까?")
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.No)
        dialog.setModal(True)
        dialog.adjustSize()
        self._center_child_window(dialog)
        return dialog

    def _center_child_window(self, child: QWidget) -> None:
        """자식 다이얼로그를 현재 메인 윈도우 화면 좌표 중앙으로 이동한다."""
        parent_rect = self.frameGeometry() if self.isVisible() else QRect(
            self.mapToGlobal(QPoint(0, 0)),
            self.size(),
        )
        child_size = child.sizeHint()
        if child_size.width() <= 0 or child_size.height() <= 0:
            child_size = child.size()
        child_rect = QRect(QPoint(0, 0), child_size)
        child_rect.moveCenter(parent_rect.center())
        child.move(child_rect.topLeft())

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
