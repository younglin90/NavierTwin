"""CFD 파일 가져오기 패널.

지원 포맷: OpenFOAM, VTK/VTU/VTP/PVD/STL/PLY, Fluent, CGNS, Gmsh, SU2.
드래그앤드롭 및 파일 브라우저 지원.

Signals:
    dataset_loaded(CFDDataset): CFD 데이터셋이 성공적으로 로드될 때 발생.
"""

from __future__ import annotations

import threading
from collections import deque
from functools import partial
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from naviertwin.core.cfd_reader import ReaderFactory


def _format_reader_extension(ext: str) -> str:
    return f"*{ext}"


def _add_info_item(info_list: QListWidget, item: tuple[object, object]) -> None:
    key, value = item
    info_list.addItem(f"{key}: {value}")


def supported_format_label() -> str:
    """고객에게 표시할 지원 포맷 요약을 반환한다."""
    return (
        "지원 포맷: OpenFOAM, VTK/VTU/VTP/PVD/STL/PLY, "
        "Fluent CAS/DAT, CGNS, Gmsh MSH, SU2"
    )


def cfd_file_filter() -> str:
    """ReaderFactory 등록 확장자를 모두 포함한 QFileDialog 필터."""
    registered = " ".join(
        map(_format_reader_extension, ReaderFactory.registered_extensions())
    )
    return (
        f"All Supported CFD ({registered});;"
        "OpenFOAM (*.foam *.OpenFOAM);;"
        "VTK / PolyData / Time-Series (*.vtk *.vtu *.vtp *.pvd *.stl *.ply);;"
        "Fluent (*.cas *.dat);;"
        "CGNS (*.cgns);;"
        "Gmsh (*.msh);;"
        "SU2 (*.su2);;"
        "All Files (*)"
    )


class ImportPanel(QWidget):
    """CFD 파일/디렉토리 가져오기 패널.

    Signals:
        dataset_loaded: 데이터 로드 완료 시 CFDDataset과 함께 발생.
    """

    dataset_loaded = Signal(object)  # CFDDataset

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._factory = ReaderFactory()
        self._current_path: Optional[Path] = None
        self._case_paths: list[Path] = []
        self._suppress_path_change_clear = False
        self._setup_ui()
        self.setAcceptDrops(True)

    # ──────────────────────────────────────────────────────────────────
    # UI 초기화
    # ──────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 제목
        title = QLabel("Import CFD Data")
        title.setObjectName("titleLabel")
        layout.addWidget(title)

        subtitle = QLabel(supported_format_label())
        subtitle.setObjectName("subtitleLabel")
        layout.addWidget(subtitle)

        # 경로 입력 그룹
        path_group = QGroupBox("파일 / 디렉토리 선택")
        path_layout = QVBoxLayout(path_group)

        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText(
            "CFD 파일 또는 OpenFOAM 케이스 디렉토리를 입력하거나 드래그하세요..."
        )
        self._path_edit.setReadOnly(False)
        path_row.addWidget(self._path_edit)

        self._browse_file_btn = QPushButton("파일 선택")
        self._browse_file_btn.setFixedWidth(90)
        self._browse_file_btn.clicked.connect(self._browse_file)
        path_row.addWidget(self._browse_file_btn)

        self._browse_dir_btn = QPushButton("폴더 선택")
        self._browse_dir_btn.setFixedWidth(90)
        self._browse_dir_btn.clicked.connect(self._browse_dir)
        path_row.addWidget(self._browse_dir_btn)

        path_layout.addLayout(path_row)

        case_row = QHBoxLayout()
        self._browse_cases_btn = QPushButton("다중 케이스 선택")
        self._browse_cases_btn.clicked.connect(self._browse_cases)
        case_row.addWidget(self._browse_cases_btn)
        self._case_label = QLabel("다중 steady-state 케이스: 선택 안 됨")
        self._case_label.setObjectName("subtitleLabel")
        case_row.addWidget(self._case_label, stretch=1)
        path_layout.addLayout(case_row)

        # 드래그앤드롭 안내
        dnd_label = QLabel("또는 파일/폴더를 여기에 드래그하세요")
        dnd_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dnd_label.setStyleSheet(
            "border: 2px dashed #4A4A70; border-radius: 6px; padding: 16px; color: #9090B0;"
        )
        dnd_label.setMinimumHeight(60)
        path_layout.addWidget(dnd_label)

        layout.addWidget(path_group)

        # 스플리터: 좌(메타데이터) | 우(미리보기)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 메타데이터 패널
        meta_group = QGroupBox("데이터셋 정보")
        meta_layout = QVBoxLayout(meta_group)

        self._info_list = QListWidget()
        self._info_list.setAlternatingRowColors(True)
        meta_layout.addWidget(self._info_list)

        splitter.addWidget(meta_group)

        # 로그 패널
        log_group = QGroupBox("로그")
        log_layout = QVBoxLayout(log_group)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(200)
        log_layout.addWidget(self._log_text)

        splitter.addWidget(log_group)
        splitter.setSizes([300, 400])
        layout.addWidget(splitter)

        # 프로그레스바
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # indeterminate
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        # 상태 레이블
        self._status_label = QLabel("파일을 선택하거나 드래그하세요.")
        self._status_label.setObjectName("subtitleLabel")
        layout.addWidget(self._status_label)

        # 하단 버튼 영역
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._clear_btn = QPushButton("초기화")
        self._clear_btn.clicked.connect(self._clear)
        btn_row.addWidget(self._clear_btn)

        self._preflight_btn = QPushButton("Readiness 점검")
        self._preflight_btn.setEnabled(False)
        self._preflight_btn.clicked.connect(self._run_preflight)
        btn_row.addWidget(self._preflight_btn)

        self._load_btn = QPushButton("데이터 로드")
        self._load_btn.setObjectName("primaryButton")
        self._load_btn.setEnabled(False)
        self._load_btn.clicked.connect(self._start_load)
        btn_row.addWidget(self._load_btn)

        layout.addLayout(btn_row)

        self._path_edit.textChanged.connect(self._on_path_changed)

    # ──────────────────────────────────────────────────────────────────
    # 드래그앤드롭
    # ──────────────────────────────────────────────────────────────────

    def dragEnterEvent(self, event: object) -> None:
        ev = event  # type: ignore[assignment]
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, event: object) -> None:
        ev = event  # type: ignore[assignment]
        urls = ev.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self._path_edit.setText(path)

    # ──────────────────────────────────────────────────────────────────
    # 슬롯
    # ──────────────────────────────────────────────────────────────────

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "CFD 파일 선택",
            "",
            cfd_file_filter(),
        )
        if path:
            self._path_edit.setText(path)

    def _browse_cases(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "다중 CFD 케이스 선택",
            "",
            cfd_file_filter(),
        )
        if not paths:
            return
        self._case_paths = list(map(Path, paths))
        self._current_path = self._case_paths[0]
        self._suppress_path_change_clear = True
        try:
            self._path_edit.setText(str(self._current_path))
        finally:
            self._suppress_path_change_clear = False
        self._case_label.setText(f"{len(self._case_paths)}개 케이스 선택됨")
        self._preflight_btn.setEnabled(True)
        self._load_btn.setEnabled(True)
        self._log(f"다중 케이스 선택: {len(self._case_paths)} files")

    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "OpenFOAM 케이스 디렉토리 선택")
        if path:
            self._path_edit.setText(path)

    def _on_path_changed(self, text: str) -> None:
        if not self._suppress_path_change_clear:
            self._case_paths = []
            self._case_label.setText("다중 steady-state 케이스: 선택 안 됨")
        p = Path(text.strip())
        self._current_path = p if text.strip() else None
        has_path = bool(text.strip())
        self._preflight_btn.setEnabled(has_path)
        self._load_btn.setEnabled(has_path)

    def _clear(self) -> None:
        self._path_edit.clear()
        self._info_list.clear()
        self._log_text.clear()
        self._status_label.setText("파일을 선택하거나 드래그하세요.")
        self._current_path = None
        self._case_paths = []
        self._case_label.setText("다중 steady-state 케이스: 선택 안 됨")
        self._preflight_btn.setEnabled(False)
        self._load_btn.setEnabled(False)

    def _run_preflight(self) -> None:
        if self._current_path is None:
            return
        self._run_preflight_path(self._current_path)

    def _run_preflight_path(self, path: Path) -> dict[str, object]:
        """CFD 입력 readiness 점검을 실행하고 결과를 로그/상태에 표시한다."""
        from naviertwin.core.validation.dataset_preflight import (
            build_dataset_preflight_report,
            format_preflight_report,
        )

        report = build_dataset_preflight_report(path)
        summary = format_preflight_report(report)
        status = str(report.get("status", "unknown"))
        score = int(report.get("readiness_score", 0) or 0)
        self._log(summary)
        self._status_label.setText(f"Preflight: {status} ({score}/100)")
        self._status_label.setObjectName("errorLabel" if status == "error" else "statusLabel")
        return report

    def _start_load(self) -> None:
        if self._current_path is None:
            return
        self._set_loading(True)
        if self._case_paths:
            self._log("Loading representative case: %s" % self._current_path)
            self._log("PhysicsNeMo case set: %d files" % len(self._case_paths))
        else:
            self._log("Loading: %s" % self._current_path)
        path = self._current_path
        case_paths = list(self._case_paths)

        def _worker() -> None:
            try:
                dataset = self._factory.create_and_read(path)
                if case_paths:
                    dataset.metadata["case_paths"] = list(map(str, case_paths))
                    dataset.metadata["case_count"] = len(case_paths)
                    dataset.metadata["case_representative"] = str(path)
                self._on_load_success(dataset)
            except Exception as exc:
                self._on_load_error(str(exc))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def _on_load_success(self, dataset: object) -> None:
        from PySide6.QtCore import QMetaObject, Qt

        # 스레드에서 Qt 메서드를 안전하게 호출
        self._pending_dataset = dataset
        QMetaObject.invokeMethod(self, "_finish_load_success", Qt.ConnectionType.QueuedConnection)

    def _on_load_error(self, msg: str) -> None:
        from PySide6.QtCore import QMetaObject, Qt

        self._pending_error = msg
        QMetaObject.invokeMethod(self, "_finish_load_error", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _finish_load_success(self) -> None:
        dataset = self._pending_dataset
        self._set_loading(False)
        self._populate_info(dataset)
        self._log("완료: %d 포인트, %d 셀, %d 타임스텝" % (
            dataset.n_points, dataset.n_cells, dataset.n_time_steps
        ))
        self._status_label.setText("로드 완료.")
        self._status_label.setObjectName("statusLabel")
        self.dataset_loaded.emit(dataset)

    @Slot()
    def _finish_load_error(self) -> None:
        msg = self._pending_error
        self._set_loading(False)
        self._log("[ERROR] " + msg)
        self._status_label.setText("로드 실패.")
        self._status_label.setObjectName("errorLabel")
        QMessageBox.critical(self, "로드 오류", msg)

    # ──────────────────────────────────────────────────────────────────
    # 헬퍼
    # ──────────────────────────────────────────────────────────────────

    def _set_loading(self, loading: bool) -> None:
        self._progress_bar.setVisible(loading)
        self._load_btn.setEnabled(not loading)
        self._preflight_btn.setEnabled(not loading and self._current_path is not None)
        self._browse_file_btn.setEnabled(not loading)
        self._browse_cases_btn.setEnabled(not loading)
        self._browse_dir_btn.setEnabled(not loading)

    def _populate_info(self, dataset: object) -> None:
        self._info_list.clear()
        items = [
            ("Points", str(dataset.n_points)),
            ("Cells", str(dataset.n_cells)),
            ("Time Steps", str(dataset.n_time_steps)),
            ("Fields", ", ".join(dataset.field_names) or "-"),
        ]
        add_item = partial(_add_info_item, self._info_list)
        deque(map(add_item, items), maxlen=0)
        deque(map(add_item, dataset.metadata.items()), maxlen=0)

    def _log(self, msg: str) -> None:
        self._log_text.append(msg)
        sb = self._log_text.verticalScrollBar()
        sb.setValue(sb.maximum())
