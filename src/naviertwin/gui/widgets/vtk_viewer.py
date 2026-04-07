"""VTK 3D 뷰어 위젯 — pyvistaqt QtInteractor 임베드.

타임스텝 슬라이더, 컬러맵 선택기, 필드 선택 드롭다운을 포함한다.

Examples:
    기본 사용법::

        from naviertwin.gui.widgets.vtk_viewer import VtkViewer
        viewer = VtkViewer(parent)
        viewer.load_dataset(cfd_dataset)
        viewer.show_field("U")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import pyvista as pv

    from naviertwin.core.cfd_reader.base import CFDDataset

_COLORMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "rainbow", "jet", "coolwarm", "RdBu", "turbo",
    "bone", "hot", "gray", "bwr",
]


class VtkViewer(QWidget):
    """PyVista QtInteractor을 감싼 3D 뷰어 위젯.

    Signals:
        timestep_changed(int): 타임스텝 인덱스가 변경될 때 발생.
        field_changed(str): 표시 필드가 변경될 때 발생.
    """

    timestep_changed = Signal(int)
    field_changed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._dataset: Optional[CFDDataset] = None
        self._current_field: str = ""
        self._current_timestep: int = 0
        self._plotter: Optional[object] = None  # pyvistaqt.QtInteractor

        self._setup_ui()

    # ──────────────────────────────────────────────────────────────────
    # UI 초기화
    # ──────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # 상단 컨트롤 바
        ctrl_bar = QWidget()
        ctrl_layout = QHBoxLayout(ctrl_bar)
        ctrl_layout.setContentsMargins(4, 2, 4, 2)
        ctrl_layout.setSpacing(8)

        # 필드 선택
        ctrl_layout.addWidget(QLabel("Field:"))
        self._field_combo = QComboBox()
        self._field_combo.setMinimumWidth(100)
        self._field_combo.currentTextChanged.connect(self._on_field_changed)
        ctrl_layout.addWidget(self._field_combo)

        # 컬러맵 선택
        ctrl_layout.addWidget(QLabel("Colormap:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(_COLORMAPS)
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        ctrl_layout.addWidget(self._cmap_combo)

        # 재설정 버튼
        self._reset_btn = QPushButton("Reset View")
        self._reset_btn.setFixedWidth(90)
        self._reset_btn.clicked.connect(self._reset_camera)
        ctrl_layout.addWidget(self._reset_btn)

        # 스크린샷 버튼
        self._screenshot_btn = QPushButton("Screenshot")
        self._screenshot_btn.setFixedWidth(90)
        self._screenshot_btn.clicked.connect(self._take_screenshot)
        ctrl_layout.addWidget(self._screenshot_btn)

        ctrl_layout.addStretch()
        layout.addWidget(ctrl_bar)

        # 3D 뷰포트 (pyvistaqt placeholder)
        self._viewport_container = QWidget()
        self._viewport_container.setMinimumSize(400, 300)
        self._viewport_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._viewport_layout = QVBoxLayout(self._viewport_container)
        self._viewport_layout.setContentsMargins(0, 0, 0, 0)

        # Placeholder 레이블 (pyvistaqt 없을 때 표시)
        self._placeholder = QLabel("3D Viewport\n(PyVista not loaded)")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #606070; font-size: 14px;")
        self._viewport_layout.addWidget(self._placeholder)

        layout.addWidget(self._viewport_container, stretch=1)

        # 하단 타임스텝 슬라이더
        ts_bar = QWidget()
        ts_layout = QHBoxLayout(ts_bar)
        ts_layout.setContentsMargins(4, 2, 4, 2)
        ts_layout.setSpacing(8)

        self._ts_label = QLabel("Step:")
        ts_layout.addWidget(self._ts_label)

        self._ts_slider = QSlider(Qt.Orientation.Horizontal)
        self._ts_slider.setMinimum(0)
        self._ts_slider.setMaximum(0)
        self._ts_slider.setValue(0)
        self._ts_slider.setEnabled(False)
        self._ts_slider.valueChanged.connect(self._on_timestep_changed)
        ts_layout.addWidget(self._ts_slider, stretch=1)

        self._ts_value_label = QLabel("0 / 0")
        self._ts_value_label.setFixedWidth(80)
        ts_layout.addWidget(self._ts_value_label)

        # 재생 버튼
        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedWidth(32)
        self._play_btn.setEnabled(False)
        self._play_btn.setCheckable(True)
        self._play_btn.toggled.connect(self._on_play_toggled)
        ts_layout.addWidget(self._play_btn)

        layout.addWidget(ts_bar)

        # 애니메이션 타이머
        from PySide6.QtCore import QTimer

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(100)  # 10 FPS
        self._anim_timer.timeout.connect(self._advance_timestep)

        self._init_plotter()

    def _init_plotter(self) -> None:
        """pyvistaqt QtInteractor 초기화 시도."""
        try:
            from pyvistaqt import QtInteractor  # type: ignore[import]

            self._plotter = QtInteractor(self._viewport_container)
            self._plotter.set_background("#1E1E2E")
            self._viewport_layout.removeWidget(self._placeholder)
            self._placeholder.hide()
            self._viewport_layout.addWidget(self._plotter)
        except Exception:
            # pyvistaqt 없거나 디스플레이 없는 환경
            self._plotter = None

    # ──────────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────────

    def load_dataset(self, dataset: CFDDataset) -> None:
        """CFDDataset을 로드하고 첫 타임스텝을 표시한다.

        Args:
            dataset: 표시할 CFDDataset.
        """
        self._dataset = dataset

        # 필드 목록 업데이트
        self._field_combo.blockSignals(True)
        self._field_combo.clear()
        self._field_combo.addItems(dataset.field_names)
        self._field_combo.blockSignals(False)

        # 타임스텝 슬라이더 업데이트
        n = max(0, dataset.n_time_steps - 1)
        self._ts_slider.setMaximum(n)
        self._ts_slider.setValue(0)
        self._ts_slider.setEnabled(n > 0)
        self._play_btn.setEnabled(n > 0)
        self._update_ts_label(0, dataset.n_time_steps)

        if dataset.field_names:
            self._current_field = dataset.field_names[0]
            self._render_current()

    def load_mesh(self, mesh: pv.DataSet, field_name: str = "") -> None:
        """단일 PyVista 메쉬를 직접 로드한다.

        Args:
            mesh: 표시할 PyVista 데이터셋.
            field_name: 컬러링에 사용할 필드 이름 (빈 문자열이면 솔리드).
        """
        if self._plotter is None:
            return
        try:
            self._plotter.clear()
            if field_name and field_name in (
                list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
            ):
                self._plotter.add_mesh(
                    mesh,
                    scalars=field_name,
                    cmap=self._cmap_combo.currentText(),
                    show_edges=False,
                )
            else:
                self._plotter.add_mesh(mesh, color="#6C7A8A", show_edges=False)
            self._plotter.reset_camera()
        except Exception:
            pass

    def show_field(self, field_name: str) -> None:
        """표시 필드를 변경한다.

        Args:
            field_name: 표시할 필드 이름.
        """
        idx = self._field_combo.findText(field_name)
        if idx >= 0:
            self._field_combo.setCurrentIndex(idx)

    def clear(self) -> None:
        """뷰포트를 초기화한다."""
        if self._plotter is not None:
            try:
                self._plotter.clear()
            except Exception:
                pass
        self._dataset = None
        self._field_combo.clear()
        self._ts_slider.setMaximum(0)
        self._ts_slider.setEnabled(False)
        self._play_btn.setEnabled(False)

    # ──────────────────────────────────────────────────────────────────
    # 내부 렌더링
    # ──────────────────────────────────────────────────────────────────

    def _render_current(self) -> None:
        """현재 타임스텝 + 필드를 렌더링한다."""
        if self._dataset is None or self._plotter is None:
            return
        try:
            mesh = self._dataset.mesh
            self._plotter.clear()
            cmap = self._cmap_combo.currentText()
            if self._current_field and self._current_field in (
                list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
            ):
                self._plotter.add_mesh(
                    mesh, scalars=self._current_field, cmap=cmap, show_edges=False
                )
            else:
                self._plotter.add_mesh(mesh, color="#6C7A8A", show_edges=False)
            self._plotter.reset_camera()
        except Exception:
            pass

    def _reset_camera(self) -> None:
        if self._plotter is not None:
            try:
                self._plotter.reset_camera()
            except Exception:
                pass

    def _take_screenshot(self) -> None:
        """스크린샷을 파일로 저장한다."""
        if self._plotter is None:
            return
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "screenshot.png", "PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            try:
                self._plotter.screenshot(path)
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────
    # 슬롯
    # ──────────────────────────────────────────────────────────────────

    def _on_field_changed(self, field_name: str) -> None:
        self._current_field = field_name
        self._render_current()
        self.field_changed.emit(field_name)

    def _on_cmap_changed(self, _cmap: str) -> None:
        self._render_current()

    def _on_timestep_changed(self, value: int) -> None:
        self._current_timestep = value
        n = self._ts_slider.maximum() + 1
        self._update_ts_label(value, n)
        self._render_current()
        self.timestep_changed.emit(value)

    def _on_play_toggled(self, checked: bool) -> None:
        if checked:
            self._play_btn.setText("⏹")
            self._anim_timer.start()
        else:
            self._play_btn.setText("▶")
            self._anim_timer.stop()

    def _advance_timestep(self) -> None:
        current = self._ts_slider.value()
        max_val = self._ts_slider.maximum()
        next_val = (current + 1) % (max_val + 1)
        self._ts_slider.setValue(next_val)

    def _update_ts_label(self, current: int, total: int) -> None:
        t_str = ""
        if self._dataset and current < len(self._dataset.time_steps):
            t_str = f" (t={self._dataset.time_steps[current]:.4g})"
        self._ts_value_label.setText(f"{current} / {max(0, total - 1)}{t_str}")

    # ──────────────────────────────────────────────────────────────────
    # 소멸자
    # ──────────────────────────────────────────────────────────────────

    def closeEvent(self, event: object) -> None:
        self._anim_timer.stop()
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:
                pass
        super().closeEvent(event)  # type: ignore[arg-type]
