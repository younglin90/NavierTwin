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
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
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
from naviertwin.gui.panels.export_panel import ExportPanel
from naviertwin.gui.panels.import_panel import ImportPanel
from naviertwin.gui.panels.model_panel import ModelPanel
from naviertwin.gui.panels.reduce_panel import ReducePanel
from naviertwin.gui.panels.twin_panel import TwinPanel
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
                f"SHA256: {result.sha256}"
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


def _load_stylesheet() -> str:
    """다크 테마 QSS를 로드한다."""
    qss_path = Path(__file__).parent / "styles" / "dark_theme.qss"
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
    ) -> None:
        super().__init__(parent)
        self._confirm_on_close = confirm_on_close
        self.setWindowTitle("NavierTwin — CFD Digital Twin")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)
        self._latest_dataset: object | None = None
        self._latest_reducer: object | None = None
        self._latest_surrogate: object | None = None
        self._latest_engine: object | None = None
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
        qss = _load_stylesheet()
        if qss:
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

        # i18n translator — default ko
        from naviertwin.utils.i18n import Translator

        self._t = Translator(lang="ko")

        self._import_panel = ImportPanel()
        self._analyze_panel = AnalyzePanel()
        self._reduce_panel = ReducePanel()
        self._model_panel = ModelPanel()
        self._twin_panel = TwinPanel()
        self._export_panel = ExportPanel()

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
        if self._postproc_panel is not None:
            self._tabs.addTab(self._postproc_panel, "⑨ Post-Tools")

        vbox.addWidget(self._tabs)

    def set_language(self, lang: str) -> None:
        """런타임 언어 전환 (탭 제목만 갱신)."""
        self._t.set_language(lang)
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

        # 도움말 메뉴
        self._help_menu = mb.addMenu("도움말(&H)")
        update_action = QAction("업데이트 확인(&U)", self)
        update_action.triggered.connect(self._check_for_updates)
        self._help_menu.addAction(update_action)
        self._help_menu.addSeparator()

        about_action = QAction("NavierTwin 정보(&A)", self)
        about_action.triggered.connect(self._show_about)
        self._help_menu.addAction(about_action)

    def _refresh_view_menu(self) -> None:
        """현재 탭 목록 전체를 보기 메뉴에 노출한다."""
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

        # Analyze 완료 상태 업데이트
        self._analyze_panel.analysis_done.connect(
            lambda method, _: self._set_status(f"분석 완료 ({method})")
        )

        # Export 완료 상태 업데이트
        self._export_panel.export_done.connect(
            lambda p: self._set_status(f"내보내기 완료: {p}")
        )
        self._export_panel.project_loaded.connect(self._on_project_loaded)

        # Twin 예측 완료
        self._twin_panel.prediction_done.connect(
            lambda _: self._set_status("예측 완료")
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
        self._latest_engine = None
        self._set_status(
            f"데이터셋 로드 완료 — {dataset.n_points} pts, "  # type: ignore[union-attr]
            f"{dataset.n_cells} cells, {dataset.n_time_steps} steps"
        )
        self._analyze_panel.set_dataset(dataset)  # type: ignore[arg-type]
        self._reduce_panel.set_dataset(dataset)    # type: ignore[arg-type]
        self._model_panel.set_dataset(dataset)     # type: ignore[arg-type]
        self._export_panel.set_dataset(dataset)    # type: ignore[arg-type]
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
        self._latest_surrogate = surrogate
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
        if engine is not None:
            self._latest_engine = engine
            self._latest_reducer = getattr(engine, "reducer", self._latest_reducer)
            self._latest_surrogate = getattr(engine, "surrogate", self._latest_surrogate)
            if self._latest_reducer is not None:
                self._model_panel.set_reducer(self._latest_reducer)
            self._twin_panel.set_engine(engine)
            self._export_panel.set_engine(engine)
            self._set_status("프로젝트 로드 완료 (dataset + TwinEngine)")
        else:
            self._set_status("프로젝트 로드 완료 (dataset)")
        loaded_meta = self._extract_project_metadata(dataset, engine)
        self._model_panel.set_loaded_metadata(loaded_meta)

    def _build_engine(self, reducer: object, surrogate: object) -> object:
        """학습된 reducer/surrogate로 TwinEngine을 구성한다."""
        from naviertwin.core.digital_twin.twin_engine import TwinEngine

        return TwinEngine.from_fitted_components(reducer, surrogate)

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

    def _open_selected_path(self, path: Path) -> None:
        """파일 메뉴 선택 경로를 포맷에 맞는 GUI 경로로 전달한다."""
        if path.suffix.lower() == ".ntwin":
            self._tabs.setCurrentWidget(self._export_panel)
            self._export_panel.load_project_path(path)
            return

        self._import_panel._path_edit.setText(str(path))
        self._tabs.setCurrentWidget(self._import_panel)

    def _save_project(self) -> None:
        self._tabs.setCurrentIndex(5)
        self._export_panel._save_project()

    def _switch_tab(self) -> None:
        action: QAction = self.sender()  # type: ignore[assignment]
        idx = action.data()
        if isinstance(idx, int):
            self._tabs.setCurrentIndex(idx)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "NavierTwin 정보",
            f"<h3>NavierTwin v{__version__}</h3>"
            "<p>CFD 후처리 결과를 AI/ROM 디지털 트윈으로 변환하는 오픈소스 툴.</p>"
            "<p>License: MIT</p>"
            "<p>Python 3.10+ | PySide6 | PyVista | PyTorch</p>",
        )

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
        QMessageBox.information(self, title, message)

    # ──────────────────────────────────────────────────────────────────
    # 헬퍼
    # ──────────────────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        self._status_label.setText(msg)

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
            event.accept()
        else:
            event.ignore()
