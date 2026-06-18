"""PyVista/pyvistaqt 기반 CFD 3D 뷰어.

AutoTessell의 안정화된 viewer 패턴을 NavierTwin 데이터 모델에 맞춰 축소한
구현이다. 실제 데스크톱 환경에서는 ``pyvistaqt.QtInteractor``를 Qt layout에
직접 임베드해 마우스 회전/줌/팬을 VTK 기본 interactor에 맡긴다. offscreen,
headless, 또는 ``NAVIERTWIN_STATIC_VIEWER=1`` 환경에서는 같은 데이터셋을
PyVista offscreen PNG로 렌더링해 테스트와 서버 환경에서도 import/load가 깨지지
않도록 한다.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from naviertwin.core.cfd_reader.base import CFDDataset


log = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    """환경변수를 boolean feature flag로 해석한다."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _qt_runtime_is_headless() -> bool:
    """Qt/PyVistaQt가 native window를 만들기 어려운 환경인지 판별한다."""
    if os.environ.get("QT_QPA_PLATFORM", "").strip().lower() == "offscreen":
        return True
    if os.name == "nt":
        return False
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _force_static_viewer_requested() -> bool:
    """VTK native window 문제를 우회하기 위한 정적 뷰어 강제 flag."""
    return _env_flag("NAVIERTWIN_STATIC_VIEWER") or _env_flag("AUTOTESSELL_STATIC_VIEWER")


try:
    import pyvista as pv

    pv.OFF_SCREEN = _qt_runtime_is_headless() or _force_static_viewer_requested()
    PYVISTA_AVAILABLE = True
except ImportError:
    pv = None  # type: ignore[assignment]
    PYVISTA_AVAILABLE = False
    log.warning("pyvista 미설치 - 3D 뷰어는 metadata shell로 동작합니다.")
except Exception as exc:  # pragma: no cover - VTK 초기화 환경 의존
    pv = None  # type: ignore[assignment]
    PYVISTA_AVAILABLE = False
    log.warning("pyvista 초기화 실패 - 3D 뷰어 비활성화: %s", exc)


if PYVISTA_AVAILABLE and not _qt_runtime_is_headless() and not _force_static_viewer_requested():
    try:
        from pyvistaqt import QtInteractor

        PYVISTAQT_AVAILABLE = True
    except ImportError:
        QtInteractor = None  # type: ignore[assignment]
        PYVISTAQT_AVAILABLE = False
        log.warning("pyvistaqt 미설치 - 정적 PNG fallback 뷰어를 사용합니다.")
    except Exception as exc:  # pragma: no cover - Qt/VTK 플랫폼 의존
        QtInteractor = None  # type: ignore[assignment]
        PYVISTAQT_AVAILABLE = False
        log.warning("pyvistaqt 초기화 실패 - 정적 PNG fallback 뷰어 사용: %s", exc)
else:
    QtInteractor = None  # type: ignore[assignment]
    PYVISTAQT_AVAILABLE = False

try:
    import shiboken6
except Exception:  # pragma: no cover - PySide6 설치 형태에 따라 달라질 수 있음
    shiboken6 = None  # type: ignore[assignment]


_COLORMAPS = ["coolwarm", "viridis", "plasma", "turbo", "jet", "rainbow", "gray"]
_SOLID_COLOR = "#c9d1d9"
_EDGE_COLOR = "#1a1f24"
_BACKGROUND_BOTTOM = "#3c4046"
_BACKGROUND_TOP = "#7a808a"


def _qt_object_is_valid(obj: Any) -> bool:
    """PySide wrapper가 살아 있고 내부 C++ QObject도 유효한지 확인한다."""
    if obj is None:
        return False
    if shiboken6 is None:
        return True
    try:
        return bool(shiboken6.isValid(obj))
    except Exception:
        return False


class VtkViewer(QWidget):
    """CFD mesh/field 표시용 3D viewer.

    Public API는 AnalyzePanel/MainWindow 연동을 위해 기존 ``VtkViewer`` 이름과
    ``load_dataset()``, ``show_field()``, ``load_field_grid_2d()`` 등을 유지한다.
    """

    timestep_changed = Signal(int)
    field_changed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._dataset: Optional[CFDDataset] = None
        self._current_field = ""
        self._current_timestep = 0
        self._plotter: Optional[Any] = None
        self._backend = "none"
        self._static_render_enabled = False
        self._static_label: Optional[QLabel] = None
        self._static_pixmap: Optional[QPixmap] = None
        self._render_scheduled = False
        self._pending_reset_camera = False
        self._last_camera_position: Optional[Any] = None
        self._last_camera_state: Optional[dict[str, Any]] = None

        self._setup_ui()
        self._init_viewer_backend()

    # ------------------------------------------------------------------
    # UI / backend
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        controls = QWidget()
        controls_layout = QGridLayout(controls)
        controls_layout.setContentsMargins(4, 2, 4, 2)
        controls_layout.setHorizontalSpacing(6)
        controls_layout.setVerticalSpacing(4)

        self._field_combo = QComboBox()
        self._field_combo.setMinimumWidth(110)
        self._field_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._field_combo.currentTextChanged.connect(self._on_field_changed)
        controls_layout.addWidget(QLabel("Field:"), 0, 0)
        controls_layout.addWidget(self._field_combo, 0, 1)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(_COLORMAPS)
        self._cmap_combo.setCurrentText("coolwarm")
        self._cmap_combo.setMinimumWidth(90)
        self._cmap_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        controls_layout.addWidget(QLabel("Cmap:"), 0, 2)
        controls_layout.addWidget(self._cmap_combo, 0, 3)

        self._edge_check = QCheckBox("Edges")
        self._edge_check.setChecked(False)
        self._edge_check.toggled.connect(self._on_edges_changed)
        controls_layout.addWidget(self._edge_check, 0, 4)

        self._iso_btn = QPushButton("ISO")
        self._front_btn = QPushButton("Front")
        self._top_btn = QPushButton("Top")
        self._side_btn = QPushButton("Side")
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setToolTip("Reset View")
        view_buttons = (
            (self._iso_btn, self._view_iso),
            (self._front_btn, self._view_front),
            (self._top_btn, self._view_top),
            (self._side_btn, self._view_side),
            (self._reset_btn, self._reset_camera),
        )
        view_index = 0
        while view_index < len(view_buttons):
            button, slot = view_buttons[view_index]
            button.setEnabled(False)
            button.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Fixed,
            )
            button.clicked.connect(slot)
            view_index += 1
        layout_buttons = (
            self._iso_btn,
            self._front_btn,
            self._top_btn,
            self._side_btn,
            self._reset_btn,
        )
        col = 0
        while col < len(layout_buttons):
            button = layout_buttons[col]
            controls_layout.addWidget(button, 1, col)
            col += 1

        self._screenshot_btn = QPushButton("Shot")
        self._screenshot_btn.setToolTip("Save Screenshot")
        self._screenshot_btn.setEnabled(False)
        self._screenshot_btn.clicked.connect(self._take_screenshot)
        controls_layout.addWidget(self._screenshot_btn, 1, 5)

        controls_layout.setColumnStretch(1, 1)
        controls_layout.setColumnStretch(3, 1)
        layout.addWidget(controls)

        self._viewport = QWidget()
        self._viewport_layout = QVBoxLayout(self._viewport)
        self._viewport_layout.setContentsMargins(0, 0, 0, 0)
        self._viewport_layout.setSpacing(0)
        self._viewport.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        layout.addWidget(self._viewport, stretch=1)

        timeline = QWidget()
        timeline_layout = QHBoxLayout(timeline)
        timeline_layout.setContentsMargins(4, 2, 4, 2)
        timeline_layout.setSpacing(8)

        timeline_layout.addWidget(QLabel("Step:"))
        self._ts_slider = QSlider(Qt.Orientation.Horizontal)
        self._ts_slider.setMinimum(0)
        self._ts_slider.setMaximum(0)
        self._ts_slider.setEnabled(False)
        self._ts_slider.valueChanged.connect(self._on_timestep_changed)
        timeline_layout.addWidget(self._ts_slider, stretch=1)

        self._ts_value_label = QLabel("0 / 0")
        self._ts_value_label.setFixedWidth(120)
        timeline_layout.addWidget(self._ts_value_label)
        layout.addWidget(timeline)

    def _init_viewer_backend(self) -> None:
        if PYVISTA_AVAILABLE and PYVISTAQT_AVAILABLE and QtInteractor is not None:
            try:
                self._plotter = QtInteractor(self._viewport)
                self._plotter.setMinimumSize(300, 240)
                self._configure_plotter()
                self._viewport_layout.addWidget(self._plotter, stretch=1)
                self._backend = "interactive"
                self._set_message("CFD 3D viewer ready.\nDrag with left mouse to rotate.")
                self._set_render_controls_enabled(True)
                log.info("NavierTwin interactive 3D viewer initialized with pyvistaqt.")
                return
            except Exception as exc:  # pragma: no cover - GUI platform dependent
                self._plotter = None
                log.warning("QtInteractor 초기화 실패 - 정적 fallback으로 전환: %s", exc)

        self._static_label = QLabel()
        self._static_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._static_label.setMinimumSize(300, 240)
        self._static_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._static_label.setStyleSheet(
            "QLabel { background-color: #0d1117; color: #c9d1d9; "
            "border-radius: 6px; padding: 8px; font-size: 13px; }"
        )
        self._viewport_layout.addWidget(self._static_label, stretch=1)
        self._backend = "static" if PYVISTA_AVAILABLE else "none"
        self._static_render_enabled = PYVISTA_AVAILABLE and not _qt_runtime_is_headless()
        if PYVISTA_AVAILABLE:
            if self._static_render_enabled:
                self._set_message(
                    "Static CFD viewer fallback.\n"
                    "Set a real display and install pyvistaqt to enable mouse rotation."
                )
            else:
                self._set_message(
                    "3D viewer loaded in headless shell mode.\n"
                    "Use a desktop display with pyvistaqt to enable interactive rendering."
                )
            self._set_render_controls_enabled(True)
        else:
            self._set_message("PyVista unavailable.\nInstall naviertwin[core] to view 3D.")
            self._set_render_controls_enabled(False)

    def _configure_plotter(self) -> None:
        if self._plotter is None:
            return
        try:
            self._plotter.set_background(_BACKGROUND_BOTTOM, top=_BACKGROUND_TOP)
        except Exception:
            try:
                self._plotter.background_color = _BACKGROUND_BOTTOM
            except Exception:
                pass
        antialias_modes = ("ssaa", "msaa", "fxaa")
        antialias_index = 0
        while antialias_index < len(antialias_modes):
            antialias = antialias_modes[antialias_index]
            try:
                self._plotter.enable_anti_aliasing(antialias)
                break
            except Exception:
                antialias_index += 1
                continue
        try:
            self._plotter.enable_lightkit()
        except Exception:
            try:
                self._plotter.enable_3_lights()
            except Exception:
                pass
        try:
            self._plotter.enable_parallel_projection()
        except Exception:
            pass
        try:
            self._plotter.enable_trackball_style()
        except Exception:
            pass
        self._show_axes()

    def _show_axes(self) -> None:
        if self._plotter is None:
            return
        try:
            self._plotter.show_axes()
        except Exception:
            try:
                self._plotter.add_axes(
                    xlabel="X",
                    ylabel="Y",
                    zlabel="Z",
                    line_width=3,
                    labels_off=False,
                )
            except Exception:
                pass

    def _set_render_controls_enabled(self, enabled: bool) -> None:
        buttons = (
            self._iso_btn,
            self._front_btn,
            self._top_btn,
            self._side_btn,
            self._reset_btn,
            self._screenshot_btn,
        )
        button_index = 0
        while button_index < len(buttons):
            button = buttons[button_index]
            button.setEnabled(enabled)
            button_index += 1
        self._cmap_combo.setEnabled(enabled)
        self._edge_check.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_dataset(self, dataset: CFDDataset) -> None:
        """CFDDataset을 로드하고 첫 필드를 표시한다."""
        self._dataset = dataset
        self._current_timestep = 0
        self.refresh_fields()

        max_step = max(0, int(getattr(dataset, "n_time_steps", 1)) - 1)
        self._ts_slider.blockSignals(True)
        self._ts_slider.setMaximum(max_step)
        self._ts_slider.setValue(0)
        self._ts_slider.setEnabled(max_step > 0)
        self._ts_slider.blockSignals(False)
        self._update_ts_label(0, max_step + 1)

        if not self._current_field:
            self._set_message(
                f"Mesh loaded: {getattr(dataset, 'n_points', 0):,} points, "
                f"{getattr(dataset, 'n_cells', 0):,} cells.\nNo scalar/vector fields."
            )
        self._schedule_render(reset_camera=True)

    def load_mesh(self, mesh: Any, field_name: str = "") -> None:
        """단일 PyVista mesh를 직접 로드한다."""
        from naviertwin.core.cfd_reader.base import CFDDataset

        field_names = self._field_names_from_mesh(mesh)
        if field_name and field_name not in field_names:
            field_names.insert(0, field_name)
        dataset = CFDDataset(
            mesh=mesh,
            time_steps=[0.0],
            field_names=field_names,
            metadata={"source": "direct_mesh"},
        )
        self.load_dataset(dataset)
        if field_name:
            self.show_field(field_name)

    def show_field(self, field_name: str) -> None:
        """표시 필드를 변경한다."""
        idx = self._field_combo.findText(field_name)
        if idx >= 0:
            self._field_combo.setCurrentIndex(idx)
        else:
            self._current_field = field_name
            self.field_changed.emit(field_name)
            self._schedule_render(reset_camera=False)

    def refresh_fields(self, prefer_field: Optional[str] = None) -> None:
        """현재 dataset의 field 목록을 콤보박스에 반영한다."""
        names = list(getattr(self._dataset, "field_names", []) or [])
        if self._dataset is not None:
            mesh_names = self._field_names_from_mesh(self._dataset.mesh)
            name_index = 0
            while name_index < len(mesh_names):
                name = mesh_names[name_index]
                if name not in names:
                    names.append(name)
                name_index += 1

        current = prefer_field or self._current_field
        self._field_combo.blockSignals(True)
        self._field_combo.clear()
        self._field_combo.addItems(names)
        if current in names:
            self._field_combo.setCurrentText(current)
            self._current_field = current
        elif names:
            preferred_idx = 0
            preferred_names = ("p", "pressure", "T", "U", "velocity")
            preferred_cursor = 0
            while preferred_cursor < len(preferred_names):
                preferred = preferred_names[preferred_cursor]
                idx = names.index(preferred) if preferred in names else -1
                if idx >= 0:
                    preferred_idx = idx
                    break
                preferred_cursor += 1
            self._field_combo.setCurrentIndex(preferred_idx)
            self._current_field = names[preferred_idx]
        else:
            self._current_field = ""
        self._field_combo.blockSignals(False)
        self._schedule_render(reset_camera=False)

    def load_field_grid_2d(self, grid: object, field_name: str = "field") -> None:
        """2D/3D 스칼라 필드를 ImageData로 래핑해 표시한다.

        Args:
            grid: ``(ny, nx)`` 또는 ``(nt, ny, nx)`` 배열.
            field_name: 표시할 필드명.
        """
        if not PYVISTA_AVAILABLE or pv is None:
            self._set_message("PyVista unavailable.\nCannot render 2D field grid.")
            return

        arr = np.asarray(grid, dtype=np.float64)
        if arr.ndim == 2:
            snapshots = arr[None, ...]
        elif arr.ndim == 3:
            snapshots = arr
        else:
            self._set_message(f"Unsupported 2D field shape: {arr.shape}")
            return

        nt, ny, nx = snapshots.shape
        image = pv.ImageData(
            dimensions=(nx, ny, 1),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )
        flattened_rows = []
        snapshot_index = 0
        while snapshot_index < len(snapshots):
            flattened_rows.append(snapshots[snapshot_index].ravel(order="F"))
            snapshot_index += 1
        flattened = np.stack(flattened_rows)
        image.point_data[field_name] = flattened[0]

        from naviertwin.core.cfd_reader.base import CFDDataset

        self.load_dataset(
            CFDDataset(
                mesh=image,
                time_steps=list(map(float, range(nt))),
                field_names=[field_name],
                metadata={
                    "source": "simulation_grid_2d",
                    "time_series_fields": {field_name: flattened},
                },
            )
        )

    def load_1d_trajectory(self, trajectory: object, field_name: str = "field") -> None:
        """1D 시공간 궤적 ``(nt, nx)``를 3D surface로 표시한다."""
        if not PYVISTA_AVAILABLE or pv is None:
            self._set_message("PyVista unavailable.\nCannot render 1D trajectory.")
            return

        arr = np.asarray(trajectory, dtype=np.float64)
        if arr.ndim != 2:
            self._set_message(f"Unsupported trajectory shape: {arr.shape}")
            return

        nt, nx = arr.shape
        x = np.arange(nx, dtype=np.float64)
        t = np.arange(nt, dtype=np.float64)
        xx, tt = np.meshgrid(x, t)
        zz = arr
        surface = pv.StructuredGrid(xx, tt, zz)
        surface.point_data[field_name] = arr.ravel(order="F")

        from naviertwin.core.cfd_reader.base import CFDDataset

        self.load_dataset(
            CFDDataset(
                mesh=surface,
                time_steps=[0.0],
                field_names=[field_name],
                metadata={"source": "simulation_trajectory_1d"},
            )
        )

    def clear(self) -> None:
        """뷰포트를 초기화한다."""
        self._dataset = None
        self._current_field = ""
        self._current_timestep = 0
        self._field_combo.clear()
        self._ts_slider.setMaximum(0)
        self._ts_slider.setValue(0)
        self._ts_slider.setEnabled(False)
        self._update_ts_label(0, 1)
        if self._plotter is not None:
            try:
                self._plotter.clear()
                self._configure_plotter()
            except Exception:
                pass
        self._set_message("CFD 3D viewer ready.\nLoad a dataset to render.")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _schedule_render(self, reset_camera: bool) -> None:
        if not _qt_object_is_valid(self):
            return
        self._pending_reset_camera = self._pending_reset_camera or reset_camera
        if self._render_scheduled:
            return
        self._render_scheduled = True
        QTimer.singleShot(0, self._render_current_pending)

    def _render_current_pending(self) -> None:
        if not _qt_object_is_valid(self):
            return
        self._render_scheduled = False
        reset_camera = self._pending_reset_camera
        self._pending_reset_camera = False
        self._render_current(reset_camera=reset_camera)

    def _render_current(self, reset_camera: bool = False) -> None:
        if self._dataset is None:
            return
        if not PYVISTA_AVAILABLE or pv is None:
            self._set_message("PyVista unavailable.\nInstall naviertwin[core] to view 3D.")
            return

        try:
            mesh, scalars = self._prepare_display_mesh()
        except Exception as exc:
            log.exception("3D display mesh preparation failed")
            self._set_message(f"3D render preparation failed:\n{exc}")
            return

        if self._backend == "interactive" and self._plotter is not None:
            self._render_interactive(mesh, scalars, reset_camera=reset_camera)
        elif self._backend == "static":
            self._render_static(mesh, scalars, reset_camera=reset_camera)

    def _prepare_display_mesh(self) -> tuple[Any, str]:
        if self._dataset is None:
            raise ValueError("dataset is not loaded")

        mesh = self._copy_mesh(self._dataset.mesh)
        scalar_name = ""
        if self._current_field:
            arr, location = self._field_array_for_current_step(mesh, self._current_field)
            if arr is not None and location is not None:
                scalar_name = self._attach_render_array(mesh, self._current_field, arr, location)

        render_mesh = self._surface_for_render(mesh)
        render_mesh = self._compute_safe_normals(render_mesh)
        return render_mesh, scalar_name

    def _render_interactive(self, mesh: Any, scalars: str, reset_camera: bool) -> None:
        if self._plotter is None:
            return

        camera_state = None
        if not reset_camera:
            camera_state = self._capture_camera_state() or self._last_camera_state

        try:
            self._plotter.clear()
            self._configure_plotter()
            self._add_mesh_to_plotter(self._plotter, mesh, scalars)
            if reset_camera:
                self._apply_camera_view("isometric", reset_camera=True)
            elif camera_state is not None:
                self._restore_camera_state(camera_state)
            self._plotter.render()
            try:
                self._last_camera_position = self._plotter.camera_position
            except Exception:
                pass
            self._last_camera_state = self._capture_camera_state()
        except Exception as exc:
            log.exception("interactive 3D render failed")
            self._set_message(f"Interactive render failed:\n{exc}")

    def _capture_camera_state(self) -> dict[str, Any] | None:
        """현재 카메라의 회전/줌 상태를 가능한 한 완전하게 저장한다."""
        if self._plotter is None:
            return None
        try:
            camera = self._plotter.camera
            return {
                "position": tuple(camera.position),
                "focal_point": tuple(camera.focal_point),
                "view_up": tuple(camera.up),
                "parallel_scale": float(camera.parallel_scale),
                "clipping_range": tuple(camera.clipping_range),
                "camera_position": self._plotter.camera_position,
            }
        except Exception:
            try:
                return {"camera_position": self._plotter.camera_position}
            except Exception:
                return None

    def _restore_camera_state(self, state: dict[str, Any]) -> None:
        """render actor 교체 후에도 사용자가 맞춘 zoom/pan/rotation을 복원한다."""
        if self._plotter is None:
            return
        try:
            camera = self._plotter.camera
            if "position" in state:
                camera.position = state["position"]
            if "focal_point" in state:
                camera.focal_point = state["focal_point"]
            if "view_up" in state:
                camera.up = state["view_up"]
            if "parallel_scale" in state:
                camera.parallel_scale = float(state["parallel_scale"])
            if "clipping_range" in state:
                camera.clipping_range = state["clipping_range"]
            return
        except Exception:
            pass
        try:
            self._plotter.camera_position = state["camera_position"]
        except Exception:
            pass

    def _render_static(self, mesh: Any, scalars: str, reset_camera: bool) -> None:
        if pv is None or not _qt_object_is_valid(self._static_label):
            return
        if not self._static_render_enabled:
            self._set_message(
                "3D data loaded.\n"
                "Static preview is disabled in headless/offscreen Qt to avoid VTK crashes."
            )
            return
        self._static_label.setText("Rendering static preview...")
        try:
            width = max(300, self._static_label.width())
            height = max(240, self._static_label.height())
            plotter = pv.Plotter(
                off_screen=True,
                window_size=(width, height),
                theme=pv.themes.DarkTheme(),
            )
            try:
                plotter.set_background(_BACKGROUND_BOTTOM, top=_BACKGROUND_TOP)
            except Exception:
                plotter.background_color = _BACKGROUND_BOTTOM
            try:
                plotter.enable_lightkit()
            except Exception:
                pass
            self._add_mesh_to_plotter(plotter, mesh, scalars)
            self._apply_static_camera(plotter, "isometric" if reset_camera else "keep")
            try:
                plotter.add_axes(xlabel="X", ylabel="Y", zlabel="Z", line_width=2)
            except Exception:
                pass
            image = plotter.screenshot(
                transparent_background=False,
                return_img=True,
            )
            try:
                plotter.close()
            except Exception:
                pass
            if image is None:
                self._set_message("Static render failed: empty screenshot.")
                return
            self._set_static_image(np.ascontiguousarray(image))
        except Exception as exc:
            log.exception("static 3D render failed")
            self._set_message(f"Static render failed:\n{exc}")

    def _add_mesh_to_plotter(self, plotter: Any, mesh: Any, scalars: str) -> None:
        kwargs: dict[str, Any] = {
            "show_edges": self._edge_check.isChecked(),
            "edge_color": _EDGE_COLOR if self._edge_check.isChecked() else None,
            "line_width": 0.6,
            "smooth_shading": True,
            "ambient": 0.30,
            "diffuse": 0.85,
            "specular": 0.18,
            "specular_power": 15,
            "lighting": True,
            "culling": "back",
            "name": "naviertwin_mesh",
        }
        if scalars:
            kwargs.update(
                {
                    "scalars": scalars,
                    "cmap": self._cmap_combo.currentText(),
                    "nan_color": "#8b949e",
                }
            )
        else:
            kwargs["color"] = _SOLID_COLOR

        actor = plotter.add_mesh(mesh, **kwargs)
        try:
            prop = actor.GetProperty()
            prop.BackfaceCullingOn()
            prop.FrontfaceCullingOff()
            prop.SetInterpolationToPhong()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _field_array_for_current_step(self, mesh: Any, field_name: str) -> tuple[np.ndarray | None, str | None]:
        if self._dataset is None:
            return None, None

        time_series = self._dataset.metadata.get("time_series_fields", {})
        if isinstance(time_series, dict) and field_name in time_series:
            arr = np.asarray(time_series[field_name])
            arr = self._select_timestep_array(arr)
            return self._normalize_field_array(arr, mesh)

        if field_name in getattr(mesh, "point_data", {}):
            return np.asarray(mesh.point_data[field_name]), "point"
        if field_name in getattr(mesh, "cell_data", {}):
            return np.asarray(mesh.cell_data[field_name]), "cell"
        return None, None

    def _select_timestep_array(self, arr: np.ndarray) -> np.ndarray:
        if self._dataset is None:
            return arr
        n_steps = max(1, int(getattr(self._dataset, "n_time_steps", 1)))
        step = min(max(0, self._current_timestep), max(0, n_steps - 1))
        n_points = int(getattr(self._dataset, "n_points", 0))
        n_cells = int(getattr(self._dataset, "n_cells", 0))

        if n_steps <= 1:
            return arr
        if arr.ndim >= 2 and arr.shape[0] == n_steps:
            return arr[step]
        if arr.ndim >= 2 and arr.shape[1] == n_steps and arr.shape[0] in {n_points, n_cells}:
            return arr[:, step]
        if arr.ndim == 1 and arr.size % n_steps == 0:
            per_step = arr.size // n_steps
            return arr[step * per_step : (step + 1) * per_step]
        if arr.ndim == 2 and arr.shape[0] in {n_points * n_steps, n_cells * n_steps}:
            per_step = arr.shape[0] // n_steps
            return arr[step * per_step : (step + 1) * per_step]
        return arr

    def _normalize_field_array(self, arr: np.ndarray, mesh: Any) -> tuple[np.ndarray | None, str | None]:
        n_points = int(getattr(mesh, "n_points", 0))
        n_cells = int(getattr(mesh, "n_cells", 0))
        values = np.asarray(arr)

        if values.ndim >= 3 and values.shape[-1] in {2, 3}:
            values = values.reshape(-1, values.shape[-1])
        elif values.ndim >= 2 and values.shape[0] not in {n_points, n_cells}:
            if values.size == n_points:
                values = values.reshape(n_points)
            elif values.size == n_cells:
                values = values.reshape(n_cells)
            elif values.shape[-1] in {2, 3} and values.size % values.shape[-1] == 0:
                values = values.reshape(-1, values.shape[-1])
            else:
                values = values.reshape(-1)

        if values.shape[0] == n_points:
            return values, "point"
        if values.shape[0] == n_cells:
            return values, "cell"
        return None, None

    def _attach_render_array(
        self,
        mesh: Any,
        field_name: str,
        arr: np.ndarray,
        location: str,
    ) -> str:
        values = np.asarray(arr)
        scalar_name = field_name
        if values.ndim > 1 and values.shape[-1] > 1:
            scalar_name = f"{field_name}__mag"
            values = np.linalg.norm(values, axis=-1)

        if location == "point":
            mesh.point_data[scalar_name] = np.asarray(values, dtype=np.float64)
        else:
            mesh.cell_data[scalar_name] = np.asarray(values, dtype=np.float64)
        return scalar_name

    def _surface_for_render(self, mesh: Any) -> Any:
        if pv is None:
            return mesh
        try:
            if isinstance(mesh, pv.UnstructuredGrid):
                return mesh.extract_surface(algorithm="dataset_surface")
        except Exception:
            try:
                if isinstance(mesh, pv.UnstructuredGrid):
                    return mesh.extract_surface()
            except Exception:
                return mesh
        return mesh

    def _compute_safe_normals(self, mesh: Any) -> Any:
        if pv is None:
            return mesh
        try:
            if isinstance(mesh, pv.PolyData) and hasattr(mesh, "compute_normals"):
                return mesh.compute_normals(
                    feature_angle=30,
                    split_vertices=True,
                    consistent_normals=True,
                    auto_orient_normals=True,
                    non_manifold_traversal=False,
                    flip_normals=False,
                )
        except Exception:
            return mesh
        return mesh

    def _copy_mesh(self, mesh: Any) -> Any:
        try:
            return mesh.copy(deep=True)
        except Exception:
            return mesh

    def _field_names_from_mesh(self, mesh: Any) -> list[str]:
        names: list[str] = []
        try:
            point_keys = tuple(mesh.point_data.keys())
            point_index = 0
            while point_index < len(point_keys):
                names.append(str(point_keys[point_index]))
                point_index += 1
            cell_keys = tuple(mesh.cell_data.keys())
            cell_index = 0
            while cell_index < len(cell_keys):
                name = str(cell_keys[cell_index])
                if name not in names:
                    names.append(name)
                cell_index += 1
        except Exception:
            pass
        return names

    # ------------------------------------------------------------------
    # Slots / controls
    # ------------------------------------------------------------------

    def _on_field_changed(self, field_name: str) -> None:
        self._current_field = field_name
        self.field_changed.emit(field_name)
        self._schedule_render(reset_camera=False)

    def _on_cmap_changed(self, _cmap: str) -> None:
        self._schedule_render(reset_camera=False)

    def _on_edges_changed(self, _checked: bool) -> None:
        self._schedule_render(reset_camera=False)

    def _on_timestep_changed(self, value: int) -> None:
        self._current_timestep = value
        self._update_ts_label(value, self._ts_slider.maximum() + 1)
        self.timestep_changed.emit(value)
        self._schedule_render(reset_camera=False)

    def _view_iso(self) -> None:
        self._apply_camera_view("isometric", reset_camera=True)

    def _view_front(self) -> None:
        self._apply_camera_view("front", reset_camera=True)

    def _view_top(self) -> None:
        self._apply_camera_view("top", reset_camera=True)

    def _view_side(self) -> None:
        self._apply_camera_view("side", reset_camera=True)

    def _reset_camera(self) -> None:
        self._apply_camera_view("isometric", reset_camera=True)

    def _apply_camera_view(self, view: str, reset_camera: bool) -> None:
        if self._backend == "interactive" and self._plotter is not None:
            try:
                if view == "front":
                    self._plotter.view_xy()
                elif view == "top":
                    self._plotter.view_xz()
                elif view == "side":
                    self._plotter.view_yz()
                else:
                    self._plotter.view_isometric()
                if reset_camera:
                    self._plotter.reset_camera()
                self._plotter.render()
                self._last_camera_position = self._plotter.camera_position
                self._last_camera_state = self._capture_camera_state()
            except Exception:
                pass
        elif self._backend == "static":
            self._schedule_render(reset_camera=True)

    @staticmethod
    def _apply_static_camera(plotter: Any, view: str) -> None:
        if view == "keep":
            view = "isometric"
        try:
            if view == "front":
                plotter.view_xy()
            elif view == "top":
                plotter.view_xz()
            elif view == "side":
                plotter.view_yz()
            else:
                plotter.view_isometric()
            plotter.reset_camera()
        except Exception:
            pass

    def _take_screenshot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
            "naviertwin_view.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg)",
        )
        if not path:
            return
        if self._backend == "interactive" and self._plotter is not None:
            try:
                self._plotter.screenshot(path, transparent_background=False)
            except Exception as exc:
                self._set_message(f"Screenshot failed:\n{exc}")
        elif self._static_pixmap is not None:
            self._static_pixmap.save(path)

    def _update_ts_label(self, current: int, total: int) -> None:
        time_text = ""
        if self._dataset is not None and current < len(getattr(self._dataset, "time_steps", [])):
            try:
                time_text = f"  t={self._dataset.time_steps[current]:.4g}"
            except Exception:
                time_text = ""
        self._ts_value_label.setText(f"{current} / {max(0, total - 1)}{time_text}")

    def _set_message(self, text: str) -> None:
        if _qt_object_is_valid(self._static_label):
            self._static_label.setText(text)

    def _set_static_image(self, image: np.ndarray) -> None:
        if not _qt_object_is_valid(self._static_label):
            return
        if image.ndim != 3 or image.shape[2] not in {3, 4}:
            self._set_message("Static render failed: unsupported screenshot format.")
            return
        height, width, channels = image.shape
        fmt = QImage.Format.Format_RGB888 if channels == 3 else QImage.Format.Format_RGBA8888
        qimage = QImage(image.data, width, height, image.strides[0], fmt).copy()
        self._static_pixmap = QPixmap.fromImage(qimage)
        self._static_label.setPixmap(
            self._static_pixmap.scaled(
                self._static_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event: Any) -> None:
        if (
            self._backend == "static"
            and _qt_object_is_valid(self._static_label)
            and self._static_pixmap
        ):
            self._static_label.setPixmap(
                self._static_pixmap.scaled(
                    self._static_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        super().resizeEvent(event)

    def closeEvent(self, event: Any) -> None:
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:
                pass
        super().closeEvent(event)


__all__ = [
    "PYVISTA_AVAILABLE",
    "PYVISTAQT_AVAILABLE",
    "VtkViewer",
    "_qt_runtime_is_headless",
]
