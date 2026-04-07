"""CFD 파일 가져오기 패널.

지원 포맷: OpenFOAM, VTK/VTU, STL. 드래그앤드롭 및 파일 브라우저 지원.

Signals:
    dataset_loaded(CFDDataset): CFD 데이터셋이 성공적으로 로드될 때 발생.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
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
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from naviertwin.core.cfd_reader.reader_factory import ReaderFactory


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

        subtitle = QLabel("지원 포맷: OpenFOAM, VTK/VTU/STL, PLY")
        subtitle.setObjectName("subtitleLabel")
        layout.addWidget(subtitle)

        # 경로 입력 그룹
        path_group = QGroupBox("파일 / 디렉토리 선택")
        path_layout = QVBoxLayout(path_group)

        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("파일 또는 OpenFOAM 케이스 디렉토리를 입력하거나 드래그하세요...")
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
            "CFD Files (*.foam *.OpenFOAM *.vtk *.vtu *.vtp *.stl *.ply);;All Files (*)",
        )
        if path:
            self._path_edit.setText(path)

    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "OpenFOAM 케이스 디렉토리 선택")
        if path:
            self._path_edit.setText(path)

    def _on_path_changed(self, text: str) -> None:
        p = Path(text.strip())
        self._current_path = p if text.strip() else None
        self._load_btn.setEnabled(bool(text.strip()))

    def _clear(self) -> None:
        self._path_edit.clear()
        self._info_list.clear()
        self._log_text.clear()
        self._status_label.setText("파일을 선택하거나 드래그하세요.")
        self._current_path = None
        self._load_btn.setEnabled(False)

    def _start_load(self) -> None:
        if self._current_path is None:
            return
        self._set_loading(True)
        self._log("Loading: %s" % self._current_path)
        path = self._current_path

        def _worker() -> None:
            try:
                dataset = self._factory.create_and_read(path)
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
        self._browse_file_btn.setEnabled(not loading)
        self._browse_dir_btn.setEnabled(not loading)

    def _populate_info(self, dataset: object) -> None:
        self._info_list.clear()
        items = [
            ("Points", str(dataset.n_points)),
            ("Cells", str(dataset.n_cells)),
            ("Time Steps", str(dataset.n_time_steps)),
            ("Fields", ", ".join(dataset.field_names) or "-"),
        ]
        for k, v in items:
            self._info_list.addItem(f"{k}: {v}")
        for mk, mv in dataset.metadata.items():
            self._info_list.addItem(f"{mk}: {mv}")

    def _log(self, msg: str) -> None:
        self._log_text.append(msg)
        sb = self._log_text.verticalScrollBar()
        sb.setValue(sb.maximum())
