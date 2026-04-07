"""내보내기 패널 — .ntwin HDF5, VTK, CSV 내보내기 및 프로젝트 저장/복원.

Signals:
    export_done(str): 내보내기 완료 시 저장된 파일 경로 발생.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from naviertwin.core.cfd_reader.base import CFDDataset


class ExportPanel(QWidget):
    """내보내기 탭 패널.

    Signals:
        export_done: 내보내기 완료 시 파일 경로와 함께 발생.
    """

    export_done = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._dataset: Optional[CFDDataset] = None
        self._engine: Optional[object] = None
        self._setup_ui()

    # ──────────────────────────────────────────────────────────────────
    # UI 초기화
    # ──────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Export")
        title.setObjectName("titleLabel")
        layout.addWidget(title)

        subtitle = QLabel("CFD 데이터, 모델, 프로젝트를 다양한 포맷으로 내보냅니다.")
        subtitle.setObjectName("subtitleLabel")
        layout.addWidget(subtitle)

        # 출력 경로
        path_group = QGroupBox("출력 경로")
        path_layout = QHBoxLayout(path_group)
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("저장할 파일 경로...")
        path_layout.addWidget(self._path_edit)
        browse_btn = QPushButton("찾기")
        browse_btn.setFixedWidth(60)
        browse_btn.clicked.connect(self._browse_path)
        path_layout.addWidget(browse_btn)
        layout.addWidget(path_group)

        # 내보내기 옵션
        options_group = QGroupBox("내보내기 옵션")
        options_form = QFormLayout(options_group)

        self._format_combo = QComboBox()
        self._format_combo.addItems([
            ".ntwin (HDF5 — 프로젝트 전체)",
            ".vtu (VTK UnstructuredGrid)",
            ".vtk (레거시 VTK)",
            ".csv (점 데이터)",
        ])
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)
        options_form.addRow("포맷:", self._format_combo)

        self._include_mesh_cb = QCheckBox("메쉬 포함")
        self._include_mesh_cb.setChecked(True)
        options_form.addRow("", self._include_mesh_cb)

        self._include_model_cb = QCheckBox("모델(TwinEngine) 포함")
        self._include_model_cb.setChecked(True)
        options_form.addRow("", self._include_model_cb)

        self._compress_cb = QCheckBox("HDF5 압축 (gzip level 4)")
        self._compress_cb.setChecked(True)
        options_form.addRow("", self._compress_cb)

        layout.addWidget(options_group)

        # 프로젝트 저장/복원
        project_group = QGroupBox("프로젝트 (.ntwin)")
        project_layout = QHBoxLayout(project_group)

        self._save_project_btn = QPushButton("프로젝트 저장")
        self._save_project_btn.clicked.connect(self._save_project)
        project_layout.addWidget(self._save_project_btn)

        self._load_project_btn = QPushButton("프로젝트 열기")
        self._load_project_btn.clicked.connect(self._load_project)
        project_layout.addWidget(self._load_project_btn)

        layout.addWidget(project_group)

        # 내보내기 버튼
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._export_btn = QPushButton("내보내기")
        self._export_btn.setObjectName("primaryButton")
        self._export_btn.clicked.connect(self._export)
        btn_row.addWidget(self._export_btn)
        layout.addLayout(btn_row)

        # 로그
        log_group = QGroupBox("로그")
        log_layout = QVBoxLayout(log_group)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        log_layout.addWidget(self._log_text)
        layout.addWidget(log_group, stretch=1)

    # ──────────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────────

    def set_dataset(self, dataset: CFDDataset) -> None:
        """내보낼 CFDDataset을 설정한다."""
        self._dataset = dataset
        self._log(f"Dataset 설정: {dataset.n_points} pts, {dataset.n_cells} cells")

    def set_engine(self, engine: object) -> None:
        """내보낼 TwinEngine을 설정한다."""
        self._engine = engine
        self._log(f"TwinEngine 설정: {type(engine).__name__}")

    # ──────────────────────────────────────────────────────────────────
    # 슬롯
    # ──────────────────────────────────────────────────────────────────

    def _browse_path(self) -> None:
        fmt_idx = self._format_combo.currentIndex()
        filters = [
            "NavierTwin Project (*.ntwin)",
            "VTK UnstructuredGrid (*.vtu)",
            "VTK Legacy (*.vtk)",
            "CSV (*.csv)",
        ]
        path, _ = QFileDialog.getSaveFileName(self, "저장 경로 선택", "", filters[fmt_idx])
        if path:
            self._path_edit.setText(path)

    def _on_format_changed(self, idx: int) -> None:
        is_hdf5 = idx == 0
        self._include_model_cb.setEnabled(is_hdf5)
        self._compress_cb.setEnabled(is_hdf5)

    def _export(self) -> None:
        path_str = self._path_edit.text().strip()
        if not path_str:
            self._log("[WARN] 출력 경로를 입력하세요.")
            return

        path = Path(path_str)
        fmt_idx = self._format_combo.currentIndex()

        try:
            if fmt_idx == 0:
                self._export_ntwin(path)
            elif fmt_idx in (1, 2):
                self._export_vtk(path)
            else:
                self._export_csv(path)
        except Exception as exc:
            self._log(f"[ERROR] 내보내기 실패: {exc}")

    def _export_ntwin(self, path: Path) -> None:
        if self._dataset is None:
            self._log("[WARN] Dataset이 없습니다.")
            return
        from naviertwin.core.export.ntwin_format import save_dataset

        compression = "gzip" if self._compress_cb.isChecked() else None
        save_dataset(self._dataset, path, compression=compression)
        self._log(f"✓ .ntwin 저장: {path}")
        self.export_done.emit(str(path))

    def _export_vtk(self, path: Path) -> None:
        if self._dataset is None:
            self._log("[WARN] Dataset이 없습니다.")
            return
        import pyvista as pv

        mesh = self._dataset.mesh
        mesh.save(str(path))
        self._log(f"✓ VTK 저장: {path}")
        self.export_done.emit(str(path))

    def _export_csv(self, path: Path) -> None:
        if self._dataset is None:
            self._log("[WARN] Dataset이 없습니다.")
            return
        import numpy as np

        mesh = self._dataset.mesh
        pts = mesh.points
        header = "x,y,z"
        rows = [pts]
        for name in self._dataset.field_names:
            if name in mesh.point_data:
                arr = np.array(mesh.point_data[name])
                if arr.ndim == 1:
                    header += f",{name}"
                    rows.append(arr.reshape(-1, 1))
                else:
                    for i in range(arr.shape[1]):
                        header += f",{name}_{i}"
                    rows.append(arr)
        data = np.hstack([r.reshape(len(pts), -1) for r in rows])
        np.savetxt(str(path), data, delimiter=",", header=header, comments="")
        self._log(f"✓ CSV 저장: {path} ({len(pts)} rows)")
        self.export_done.emit(str(path))

    def _save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "프로젝트 저장", "project.ntwin", "NavierTwin Project (*.ntwin)"
        )
        if not path:
            return
        if self._dataset is None:
            self._log("[WARN] Dataset이 없습니다.")
            return
        try:
            self._path_edit.setText(path)
            self._export_ntwin(Path(path))
        except Exception as exc:
            self._log(f"[ERROR] 프로젝트 저장 실패: {exc}")

    def _load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "프로젝트 열기", "", "NavierTwin Project (*.ntwin)"
        )
        if not path:
            return
        try:
            from naviertwin.core.export.ntwin_format import load_dataset
            dataset = load_dataset(Path(path))
            self._dataset = dataset
            self._log(f"✓ 프로젝트 로드: {path} ({dataset.n_points} pts)")
        except Exception as exc:
            self._log(f"[ERROR] 프로젝트 로드 실패: {exc}")

    def _log(self, msg: str) -> None:
        self._log_text.append(msg)
        sb = self._log_text.verticalScrollBar()
        sb.setValue(sb.maximum())
