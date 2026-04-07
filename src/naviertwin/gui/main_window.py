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

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
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

from naviertwin.gui.panels.analyze_panel import AnalyzePanel
from naviertwin.gui.panels.export_panel import ExportPanel
from naviertwin.gui.panels.import_panel import ImportPanel
from naviertwin.gui.panels.model_panel import ModelPanel
from naviertwin.gui.panels.reduce_panel import ReducePanel
from naviertwin.gui.panels.twin_panel import TwinPanel


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

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("NavierTwin — CFD Digital Twin")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)

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

        self._import_panel = ImportPanel()
        self._analyze_panel = AnalyzePanel()
        self._reduce_panel = ReducePanel()
        self._model_panel = ModelPanel()
        self._twin_panel = TwinPanel()
        self._export_panel = ExportPanel()

        self._tabs.addTab(self._import_panel,  "① Import")
        self._tabs.addTab(self._analyze_panel, "② Analyze")
        self._tabs.addTab(self._reduce_panel,  "③ Reduce")
        self._tabs.addTab(self._model_panel,   "④ Model")
        self._tabs.addTab(self._twin_panel,    "⑤ Twin")
        self._tabs.addTab(self._export_panel,  "⑥ Export")

        vbox.addWidget(self._tabs)

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
        view_menu = mb.addMenu("보기(&V)")
        for i, name in enumerate(["Import", "Analyze", "Reduce", "Model", "Twin", "Export"]):
            action = QAction(f"{name} 탭", self)
            action.setShortcut(f"Ctrl+{i+1}")
            action.setData(i)
            action.triggered.connect(self._switch_tab)
            view_menu.addAction(action)

        # 도움말 메뉴
        help_menu = mb.addMenu("도움말(&H)")
        about_action = QAction("NavierTwin 정보(&A)", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_statusbar(self) -> None:
        sb: QStatusBar = self.statusBar()

        self._status_label = QLabel("준비")
        sb.addWidget(self._status_label, stretch=1)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFixedWidth(200)
        self._progress_bar.setVisible(False)
        sb.addPermanentWidget(self._progress_bar)

        version_label = QLabel("v1.0.0")
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

        # Export 완료 상태 업데이트
        self._export_panel.export_done.connect(
            lambda p: self._set_status(f"내보내기 완료: {p}")
        )

        # Twin 예측 완료
        self._twin_panel.prediction_done.connect(
            lambda _: self._set_status("예측 완료")
        )

    # ──────────────────────────────────────────────────────────────────
    # 시그널 핸들러
    # ──────────────────────────────────────────────────────────────────

    def _on_dataset_loaded(self, dataset: object) -> None:
        self._set_status(
            f"데이터셋 로드 완료 — {dataset.n_points} pts, "  # type: ignore[union-attr]
            f"{dataset.n_cells} cells, {dataset.n_time_steps} steps"
        )
        self._analyze_panel.set_dataset(dataset)  # type: ignore[arg-type]
        self._reduce_panel.set_dataset(dataset)    # type: ignore[arg-type]
        self._export_panel.set_dataset(dataset)    # type: ignore[arg-type]
        # Import 탭 완료 후 Analyze 탭으로 자동 이동
        self._tabs.setCurrentIndex(1)

    def _on_reduction_done(self, method: str, reducer: object) -> None:
        self._set_status(f"차원 축소 완료 ({method})")
        self._model_panel.set_reducer(reducer)
        self._tabs.setCurrentIndex(3)

    def _on_model_trained(self, model_type: str, surrogate: object) -> None:
        self._set_status(f"모델 학습 완료 ({model_type})")
        self._tabs.setCurrentIndex(4)

    # ──────────────────────────────────────────────────────────────────
    # 메뉴 액션
    # ──────────────────────────────────────────────────────────────────

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "CFD 파일 열기",
            "",
            "All CFD Files (*.foam *.OpenFOAM *.vtk *.vtu *.vtp *.stl *.ntwin);;"
            "NavierTwin Project (*.ntwin);;"
            "OpenFOAM (*.foam *.OpenFOAM);;"
            "VTK (*.vtk *.vtu *.vtp *.stl);;All Files (*)",
        )
        if path:
            self._import_panel._path_edit.setText(path)
            self._tabs.setCurrentIndex(0)

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
            "<h3>NavierTwin v1.0.0</h3>"
            "<p>CFD 후처리 결과를 AI/ROM 디지털 트윈으로 변환하는 오픈소스 툴.</p>"
            "<p>License: GPL-3.0</p>"
            "<p>Python 3.10+ | PySide6 | PyVista | PyTorch</p>",
        )

    # ──────────────────────────────────────────────────────────────────
    # 헬퍼
    # ──────────────────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        self._status_label.setText(msg)

    def closeEvent(self, event: QCloseEvent) -> None:
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
