"""유동 분석 패널 — Q-criterion, FFT/PSD, y+ 계산 및 결과 시각화.

Signals:
    analysis_done(str, object): 분석 완료 시 (분석 이름, 결과 메쉬) 발생.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from naviertwin.core.cfd_reader.base import CFDDataset


# 정적 분석(단일 스냅샷에서도 동작) → 그 후 시계열 분석.
# 시계열 분석은 단일 스냅샷일 때 비활성화되며, 사용자가 한눈에 구분할 수
# 있도록 명단 하단에 모아둔다.
_ANALYSIS_METHODS: list[tuple[str, str]] = [
    # ── 정적 분석(단일 스냅샷 OK) ──
    ("Q-criterion", "q_criterion"),
    ("λ₂ Criterion", "lambda2"),
    ("y+ (Wall Units)", "yplus"),
    ("해석해 비교 (Analytic)", "analytic"),
    ("Boundary Layer Thickness", "boundary_layer"),
    ("Nondimensional Numbers", "nondim"),
    ("FTLE / LCS Quick Check", "ftle"),
    ("PGD 3D Quick Decomposition", "pgd"),
    ("Entropy Generation 2D", "entropy_generation"),
    # ── 시계열 분석(스냅샷 ≥ 2 필요) ──
    ("FFT / PSD", "fft_psd"),
    ("SPOD (Modal)", "spod"),
    ("SINDy (Equation Discovery)", "sindy"),
    ("Wavelet / STFT", "wavelet"),
]

# 시간 스냅샷이 ≥2 일 때만 의미 있는 분석 (단일 스냅샷이면 비활성화).
_TIME_SERIES_METHODS: frozenset[str] = frozenset(
    {"fft_psd", "spod", "sindy", "wavelet"}
)

# 분석 결과를 3D viewer 에 노출할 필드 매핑 (mesh.point_data 키).
_METHOD_RESULT_FIELDS: dict[str, str] = {
    "q_criterion": "Q-criterion",
    "lambda2": "lambda2",
    "yplus": "yplus",
}


def analysis_method_labels() -> list[str]:
    """Analyze 탭에 표시되는 분석 방법 이름 목록을 반환한다."""
    return [label for label, _ in _ANALYSIS_METHODS]


def analysis_result_field(method: str) -> str | None:
    """분석 method가 mesh에 추가하는 viewer field 이름을 반환한다."""
    return _METHOD_RESULT_FIELDS.get(method)


class AnalyzePanel(QWidget):
    """유동 분석 탭 패널.

    Q-criterion / λ₂, FFT/PSD, y+ 계산 기능을 제공한다.

    Signals:
        analysis_done: 분석 완료 시 결과 메쉬와 함께 발생.
    """

    analysis_done = Signal(str, object)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        use_embedded_viewer: bool = True,
    ) -> None:
        super().__init__(parent)
        self._dataset: Optional[CFDDataset] = None
        self._use_embedded_viewer = use_embedded_viewer
        self._vtk_viewer: object | None = None
        self._setup_ui()

    # ──────────────────────────────────────────────────────────────────
    # UI 초기화
    # ──────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 좌측: 분석 목록 + 파라미터
        left = QWidget()
        left.setFixedWidth(280)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        title = QLabel("Flow Analysis")
        title.setObjectName("titleLabel")
        left_layout.addWidget(title)

        # 분석 선택
        method_group = QGroupBox("분석 방법")
        method_layout = QVBoxLayout(method_group)
        self._method_list = QListWidget()
        self._method_list.addItems(analysis_method_labels())
        self._method_list.currentRowChanged.connect(self._on_method_selected)
        method_layout.addWidget(self._method_list)
        left_layout.addWidget(method_group)

        # 파라미터 스택
        param_group = QGroupBox("파라미터")
        param_layout = QVBoxLayout(param_group)
        self._param_stack = QStackedWidget()

        # _ANALYSIS_METHODS 의 row 순서와 1:1 정합되어야 한다.
        # ── 정적 분석 ──
        self._param_stack.addWidget(self._build_qcrit_params())   # 0 q_criterion
        self._param_stack.addWidget(self._build_qcrit_params())   # 1 lambda2
        self._param_stack.addWidget(self._build_yplus_params())   # 2 yplus
        self._param_stack.addWidget(self._build_analytic_params())  # 3 analytic
        self._param_stack.addWidget(self._build_info_params(      # 4 boundary_layer
            "경계층 두께는 y 좌표와 첫 번째 속도 필드 프로파일을 사용합니다."))
        self._param_stack.addWidget(self._build_info_params(      # 5 nondim
            "무차원수는 표준 공기/길이 기본값으로 계산합니다."))
        self._param_stack.addWidget(self._build_info_params(      # 6 ftle
            "FTLE는 내장 2D 비정상 유동 quick check를 실행합니다."))
        self._param_stack.addWidget(self._build_info_params(      # 7 pgd
            "PGD는 대표 3D 텐서 quick decomposition을 실행합니다."))
        self._param_stack.addWidget(self._build_info_params(      # 8 entropy_generation
            "엔트로피 생성률은 2D 열유동 quick check를 실행합니다."))
        # ── 시계열 분석 ──
        self._param_stack.addWidget(self._build_fft_params())     # 9 fft_psd
        self._param_stack.addWidget(self._build_info_params(      # 10 spod
            "SPOD는 첫 번째 시계열 필드로 실행합니다."))
        self._param_stack.addWidget(self._build_sindy_params())   # 11 sindy
        self._param_stack.addWidget(self._build_info_params(      # 12 wavelet
            "Wavelet/STFT는 대표 신호를 시간-주파수로 분석합니다."))

        param_layout.addWidget(self._param_stack)
        left_layout.addWidget(param_group)

        # 실행 버튼
        self._run_btn = QPushButton("분석 실행")
        self._run_btn.setObjectName("primaryButton")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run_analysis)
        left_layout.addWidget(self._run_btn)

        left_layout.addStretch()
        layout.addWidget(left)

        # 우측: 3D viewer + 결과 로그 + 해석해 비교 (탭으로 분리)
        right_splitter = QSplitter()
        right_splitter.setOrientation(Qt.Orientation.Vertical)

        # 시각화 탭 — CFDDataset mesh/field를 PyVista viewer에 직접 표시한다.
        from naviertwin.gui.widgets.analytic_compare_widget import (
            AnalyticCompareWidget,
        )
        self._viewer_tabs = QTabWidget()
        self._compare_widget = AnalyticCompareWidget()
        if self._use_embedded_viewer:
            from naviertwin.gui.widgets.vtk_viewer import VtkViewer

            self._vtk_viewer = VtkViewer()
            self._viewer_tabs.addTab(self._vtk_viewer, "3D 뷰어")
        else:
            viewer_hint = QLabel(
                "3D 뷰어는 오른쪽 전역 패널에 항상 표시됩니다.\n"
                "탭을 이동해도 현재 mesh/field와 예측 결과가 유지됩니다."
            )
            viewer_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            viewer_hint.setObjectName("subtitleLabel")
            self._viewer_tabs.addTab(viewer_hint, "3D 뷰어")
        self._viewer_tabs.addTab(self._compare_widget, "해석해 비교")
        right_splitter.addWidget(self._viewer_tabs)

        # 결과 로그
        log_group = QGroupBox("결과 / 로그")
        log_layout = QVBoxLayout(log_group)
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        log_layout.addWidget(self._result_text)
        right_splitter.addWidget(log_group)

        # 상태 레이블
        self._status_label = QLabel("데이터를 먼저 가져오세요.")
        self._status_label.setObjectName("subtitleLabel")
        right_splitter.addWidget(self._status_label)

        right_splitter.setSizes([520, 180, 40])
        layout.addWidget(right_splitter, stretch=1)

        self._method_list.setCurrentRow(0)

    def _build_qcrit_params(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(4, 4, 4, 4)
        combo = QComboBox()
        combo.setObjectName("velocity_combo")
        combo.addItems(["U", "velocity", "u"])
        form.addRow("속도 필드:", combo)
        return w

    def _build_fft_params(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(4, 4, 4, 4)
        dt_spin = QDoubleSpinBox()
        dt_spin.setRange(1e-6, 1e3)
        dt_spin.setValue(0.01)
        dt_spin.setDecimals(6)
        form.addRow("Δt (s):", dt_spin)
        combo = QComboBox()
        combo.addItems(["U", "p", "T"])
        form.addRow("필드:", combo)
        return w

    def _build_analytic_params(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(4, 4, 4, 4)

        flow_combo = QComboBox()
        flow_combo.setObjectName("analytic_flow_combo")
        flow_combo.addItems(["Couette", "Poiseuille 2D", "Poiseuille Pipe"])
        form.addRow("유동 유형:", flow_combo)

        field_combo = QComboBox()
        field_combo.setObjectName("analytic_field_combo")
        field_combo.addItems(["U"])
        form.addRow("비교 필드:", field_combo)

        axis_combo = QComboBox()
        axis_combo.setObjectName("analytic_axis_combo")
        axis_combo.addItems(["y", "x", "z"])
        form.addRow("샘플 축:", axis_combo)

        # 공용 파라미터
        param1 = QDoubleSpinBox()
        param1.setObjectName("analytic_param1")
        param1.setRange(-1e9, 1e9)
        param1.setDecimals(6)
        param1.setValue(1.0)
        form.addRow("U_top / dp/dx:", param1)

        mu_spin = QDoubleSpinBox()
        mu_spin.setObjectName("analytic_mu")
        mu_spin.setRange(1e-10, 1e6)
        mu_spin.setDecimals(8)
        mu_spin.setValue(1.0)
        form.addRow("μ (Pa·s):", mu_spin)

        h_spin = QDoubleSpinBox()
        h_spin.setObjectName("analytic_h")
        h_spin.setRange(1e-6, 1e6)
        h_spin.setDecimals(6)
        h_spin.setValue(1.0)
        form.addRow("H / R (m):", h_spin)

        n_spin = QDoubleSpinBox()
        n_spin.setObjectName("analytic_n")
        n_spin.setRange(5, 1000)
        n_spin.setDecimals(0)
        n_spin.setValue(50)
        form.addRow("샘플 수:", n_spin)

        return w

    def _build_yplus_params(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(4, 4, 4, 4)

        wss_combo = QComboBox()
        wss_combo.addItems(["wallShearStress", "tau_w"])
        form.addRow("벽면 전단응력 필드:", wss_combo)

        rho_spin = QDoubleSpinBox()
        rho_spin.setDecimals(4)
        rho_spin.setRange(0.001, 1e5)
        rho_spin.setValue(1.225)
        form.addRow("밀도 ρ (kg/m³):", rho_spin)

        # 주의: setDecimals 는 반드시 setValue 이전에 호출되어야 한다 — 아니면
        # 작은 값(1.5e-5)이 기본 2 decimals 로 0.00 으로 반올림된다.
        nu_spin = QDoubleSpinBox()
        nu_spin.setDecimals(8)
        nu_spin.setRange(1e-10, 1.0)
        nu_spin.setSingleStep(1e-6)
        nu_spin.setValue(1.5e-5)
        form.addRow("동점성계수 ν (m²/s):", nu_spin)

        y_spin = QDoubleSpinBox()
        y_spin.setDecimals(8)
        y_spin.setRange(1e-10, 1.0)
        y_spin.setSingleStep(1e-5)
        y_spin.setValue(1e-4)
        form.addRow("첫 번째 셀 높이 y (m):", y_spin)

        return w

    def _build_info_params(self, text: str) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        label = QLabel(text)
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addStretch()
        return w

    def _build_sindy_params(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(4, 4, 4, 4)

        degree_spin = QSpinBox()
        degree_spin.setObjectName("sindy_degree")
        degree_spin.setRange(1, 5)
        degree_spin.setValue(2)
        form.addRow("Polynomial degree:", degree_spin)

        threshold_spin = QDoubleSpinBox()
        threshold_spin.setObjectName("sindy_threshold")
        threshold_spin.setRange(0.0, 10.0)
        threshold_spin.setDecimals(4)
        threshold_spin.setSingleStep(0.01)
        threshold_spin.setValue(0.05)
        form.addRow("Threshold:", threshold_spin)

        return w

    # ──────────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────────

    def set_dataset(self, dataset: CFDDataset) -> None:
        """분석할 CFDDataset을 설정한다."""
        self._dataset = dataset
        self._run_btn.setEnabled(True)
        self._status_label.setText(
            f"데이터셋 준비 완료 ({dataset.n_points} points, {dataset.n_time_steps} steps)"
        )
        # 속도 필드 콤보 업데이트 — 가능하면 U/velocity/u 를 기본 선택.
        velocity_priority = ("U", "velocity", "u", "Velocity")
        for page_idx in [self._method_index("q_criterion"), self._method_index("lambda2")]:
            page = self._param_stack.widget(page_idx)
            combo = page.findChild(QComboBox, "velocity_combo")
            if combo is not None:
                combo.clear()
                combo.addItems(dataset.field_names)
                for name in velocity_priority:
                    idx = combo.findText(name)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                        break

        # 해석해 비교 필드 콤보 업데이트
        analytic_page = self._param_stack.widget(self._method_index("analytic"))
        if analytic_page is not None:
            field_combo = analytic_page.findChild(QComboBox, "analytic_field_combo")
            if field_combo is not None and dataset.field_names:
                field_combo.clear()
                field_combo.addItems(dataset.field_names)

        # 시간 스냅샷이 부족하면 시계열 분석 비활성화.
        self._update_method_availability(dataset.n_time_steps)

        # viewer 에 데이터셋을 전달해 mesh/field를 표시한다.
        if self._vtk_viewer is not None:
            try:
                self._vtk_viewer.load_dataset(dataset)
            except Exception:
                pass

    def _update_method_availability(self, n_time_steps: int) -> None:
        """단일 스냅샷일 때 시계열 의존 분석 항목을 disable + 흐린 글씨."""
        from PySide6.QtGui import QBrush, QColor, QFont

        has_time = int(n_time_steps) > 1
        disabled_color = QBrush(QColor("#606070"))   # 흐린 회색
        enabled_color = QBrush(QColor("#E0E0E0"))    # 일반 텍스트
        for row, (label, key) in enumerate(_ANALYSIS_METHODS):
            item: QListWidgetItem = self._method_list.item(row)
            if item is None:
                continue
            requires_time = key in _TIME_SERIES_METHODS
            font: QFont = item.font()
            if requires_time and not has_time:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setForeground(disabled_color)
                font.setItalic(True)
                item.setFont(font)
                item.setToolTip(
                    "이 분석은 시계열 데이터(스냅샷 ≥ 2)가 필요합니다."
                )
            else:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled)
                item.setForeground(enabled_color)
                font.setItalic(False)
                item.setFont(font)
                item.setToolTip("")
        # 비활성 항목이 현재 선택된 경우 첫 활성 항목으로 이동.
        cur = self._method_list.currentItem()
        if cur is not None and not (cur.flags() & Qt.ItemFlag.ItemIsEnabled):
            for row in range(self._method_list.count()):
                if self._method_list.item(row).flags() & Qt.ItemFlag.ItemIsEnabled:
                    self._method_list.setCurrentRow(row)
                    break

    # ──────────────────────────────────────────────────────────────────
    # 슬롯
    # ──────────────────────────────────────────────────────────────────

    def _on_method_selected(self, row: int) -> None:
        self._param_stack.setCurrentIndex(row)

    def _run_analysis(self) -> None:
        if self._dataset is None:
            return
        row = self._method_list.currentRow()
        method = _ANALYSIS_METHODS[row][1] if row >= 0 else "q_criterion"
        try:
            result = self._dispatch(method)
            self._result_text.append(f"[{method}] 완료\n{result}\n")
            self.analysis_done.emit(method, result)
            if method == "analytic" and isinstance(result, dict):
                self._compare_widget.update_result(result)
                # 비교 차트가 있는 탭으로 자동 전환.
                if hasattr(self, "_viewer_tabs") and self._viewer_tabs is not None:
                    self._viewer_tabs.setCurrentWidget(self._compare_widget)
            else:
                # viewer 에 분석 결과 필드를 동기화한다.
                new_field = _METHOD_RESULT_FIELDS.get(method)
                if new_field is not None:
                    self._sync_result_field_to_viewer(new_field)
        except Exception as exc:
            self._result_text.append(f"[ERROR] {method}: {exc}\n")

    def _sync_result_field_to_viewer(self, field_name: str) -> None:
        """mesh.point_data 에 추가된 분석 결과 필드를 viewer 에 노출."""
        if self._dataset is None or self._vtk_viewer is None:
            return
        try:
            mesh = self._dataset.mesh
            if field_name not in mesh.point_data and field_name not in mesh.cell_data:
                return
        except Exception:
            return
        # dataset.field_names 에도 반영.
        try:
            if field_name not in self._dataset.field_names:
                self._dataset.field_names.append(field_name)
        except Exception:
            pass
        # viewer 콤보 갱신 + 새 필드 선택.
        self._vtk_viewer.refresh_fields(prefer_field=field_name)
        # viewer 탭으로 자동 전환.
        if hasattr(self, "_viewer_tabs") and self._viewer_tabs is not None:
            self._viewer_tabs.setCurrentWidget(self._vtk_viewer)

    def _dispatch(self, method: str) -> object:
        from naviertwin.core.flow_analysis.vortex.q_criterion import (
            compute_lambda2,
            compute_q_criterion,
        )

        mesh = self._dataset.mesh  # type: ignore[union-attr]

        if method == "q_criterion":
            page = self._param_stack.widget(self._method_index("q_criterion"))
            combo = page.findChild(QComboBox, "velocity_combo")
            vel = combo.currentText() if combo else "U"
            result_mesh = compute_q_criterion(mesh, vel)
            vals = result_mesh.point_data.get("Q-criterion")
            if vals is not None:
                # 원본 mesh 에도 결과 필드 복사 → 뷰어 컨투어 표시 가능.
                try:
                    mesh.point_data["Q-criterion"] = vals
                except Exception:
                    pass
                return f"Q range: [{vals.min():.4g}, {vals.max():.4g}]"
            return result_mesh

        elif method == "lambda2":
            page = self._param_stack.widget(self._method_index("lambda2"))
            combo = page.findChild(QComboBox, "velocity_combo")
            vel = combo.currentText() if combo else "U"
            result_mesh = compute_lambda2(mesh, vel)
            vals = result_mesh.point_data.get("lambda2")
            if vals is not None:
                try:
                    mesh.point_data["lambda2"] = vals
                except Exception:
                    pass
                return f"λ₂ range: [{vals.min():.4g}, {vals.max():.4g}]"
            return result_mesh

        elif method == "fft_psd":
            import numpy as np

            from naviertwin.core.flow_analysis.statistics.fft_psd import (
                compute_fft,
                find_dominant_frequencies,
            )

            n_steps = int(self._dataset.n_time_steps)  # type: ignore[union-attr]
            if n_steps <= 1:
                return "FFT: time-series 데이터가 없어 실행할 수 없습니다."

            page = self._param_stack.widget(self._method_index("fft_psd"))
            spins = page.findChildren(QDoubleSpinBox)
            dt = spins[0].value() if spins else 0.01
            field = self._dataset.field_names[0] if self._dataset.field_names else None  # type: ignore[union-attr]
            if field:
                snapshots = self._dataset.extract_field_snapshots(field)  # type: ignore[union-attr]
                if snapshots.shape[1] <= 1:
                    return "FFT: 현재 데이터 구조에서는 시계열을 복원할 수 없습니다."
                signal = snapshots.mean(axis=0).astype(float)

                freqs, amps = compute_fft(signal, dt)
                peaks = find_dominant_frequencies(freqs, amps, n_peaks=3)
                peak_text = [f"{p['frequency']:.4g} Hz" for p in peaks]
                return f"Top frequencies: {peak_text}"
            return "FFT: 필드 없음"

        elif method == "yplus":
            import numpy as np

            from naviertwin.core.flow_analysis.boundary_layer.yplus import compute_yplus

            page = self._param_stack.widget(self._method_index("yplus"))
            spins = page.findChildren(QDoubleSpinBox)
            rho = spins[0].value() if len(spins) > 0 else 1.225
            nu = spins[1].value() if len(spins) > 1 else 1.5e-5
            y_wall = spins[2].value() if len(spins) > 2 else 1e-4

            # 벽면 전단응력 필드 탐색
            wss_combo = page.findChild(QComboBox)
            wss_field = wss_combo.currentText() if wss_combo else "wallShearStress"
            if wss_field in mesh.point_data:
                tau = mesh.point_data[wss_field]
                tau_arr = np.asarray(tau, dtype=float)
                if tau_arr.ndim == 1:
                    # 크기만 있는 경우에도 각 포인트별 전단응력으로 계산되도록 2D로 맞춘다.
                    tau_arr = tau_arr[:, np.newaxis]
                yplus = compute_yplus(
                    tau_arr,
                    rho,
                    nu,
                    np.full(tau_arr.shape[0], y_wall, dtype=float),
                )
                # 원본 mesh 에도 yplus 필드 추가 → 뷰어 컨투어 표시.
                try:
                    mesh.point_data["yplus"] = np.asarray(yplus, dtype=float)
                except Exception:
                    pass
                return f"y+ range: [{yplus.min():.4g}, {yplus.max():.4g}], mean={yplus.mean():.4g}"
            return "y+: 벽면 전단응력 필드 없음"

        elif method == "analytic":
            return self._run_analytic_compare(mesh)

        elif method == "spod":
            return self._run_spod()

        elif method == "sindy":
            return self._run_sindy()

        elif method == "wavelet":
            return self._run_wavelet()

        elif method == "boundary_layer":
            return self._run_boundary_layer()

        elif method == "nondim":
            return self._run_nondim()

        elif method == "ftle":
            return self._run_ftle()

        elif method == "pgd":
            return self._run_pgd()

        elif method == "entropy_generation":
            return self._run_entropy_generation()

        return "알 수 없는 분석 방법"

    def _run_analytic_compare(self, mesh: object) -> object:
        """해석해와 수치해를 비교하고 결과 dict 를 반환한다."""
        import numpy as np

        from naviertwin.core.validation.analytic_solutions import (
            compare_against_analytic,
            couette_flow,
            poiseuille_flow_2d,
            poiseuille_pipe,
        )

        page = self._param_stack.widget(self._method_index("analytic"))
        flow_combo: QComboBox = page.findChild(QComboBox, "analytic_flow_combo")
        field_combo: QComboBox = page.findChild(QComboBox, "analytic_field_combo")
        axis_combo: QComboBox = page.findChild(QComboBox, "analytic_axis_combo")
        p1: QDoubleSpinBox = page.findChild(QDoubleSpinBox, "analytic_param1")
        mu: QDoubleSpinBox = page.findChild(QDoubleSpinBox, "analytic_mu")
        h: QDoubleSpinBox = page.findChild(QDoubleSpinBox, "analytic_h")
        n: QDoubleSpinBox = page.findChild(QDoubleSpinBox, "analytic_n")

        flow = flow_combo.currentText() if flow_combo else "Couette"
        field = field_combo.currentText() if field_combo else "U"
        axis = axis_combo.currentText() if axis_combo else "y"
        param1_val = p1.value() if p1 else 1.0
        mu_val = mu.value() if mu else 1.0
        h_val = h.value() if h else 1.0
        n_val = int(n.value()) if n else 50

        coords = np.linspace(0.0, h_val, n_val)
        if flow == "Couette":
            sol = couette_flow(U_top=param1_val, H=h_val, y=coords)
        elif flow == "Poiseuille 2D":
            sol = poiseuille_flow_2d(dpdx=param1_val, mu=mu_val, H=h_val, y=coords)
        else:  # Poiseuille Pipe
            sol = poiseuille_pipe(dpdx=param1_val, mu=mu_val, R=h_val, r=coords)

        return compare_against_analytic(mesh, sol, field_name=field, axis=axis)

    def _run_spod(self) -> str:
        """첫 번째 시계열 필드로 SPOD quick diagnostic을 실행한다."""
        import numpy as np

        from naviertwin.core.flow_analysis.modal.spod import compute_spod

        snapshots = self._field_snapshots()
        if snapshots.shape[1] < 8:
            return "SPOD: 최소 8개 이상의 시간 스냅샷이 필요합니다."

        n_fft = max(8, min(32, snapshots.shape[1]))
        result = compute_spod(snapshots, dt=self._time_step(), n_fft=n_fft, n_modes=3)
        eig = np.asarray(result["eigenvalues"], dtype=float)
        return (
            f"SPOD: n_freq={len(result['frequencies'])}, "
            f"n_modes={eig.shape[1]}, leading_energy={eig[0, 0]:.4g}"
        )

    def _run_sindy(self) -> str:
        """대표 시계열 신호로 SINDy equation discovery를 실행한다."""
        import numpy as np

        from naviertwin.core.flow_analysis.modal.sindy_wrapper import SINDy

        signal = np.asarray(self._representative_signal(), dtype=float).ravel()
        if signal.size < 5:
            return "SINDy: 최소 5개 이상의 시간 샘플이 필요합니다."

        page = self._param_stack.widget(self._method_index("sindy"))
        degree: QSpinBox = page.findChild(QSpinBox, "sindy_degree")
        threshold: QDoubleSpinBox = page.findChild(QDoubleSpinBox, "sindy_threshold")
        poly_degree = degree.value() if degree else 2
        threshold_value = threshold.value() if threshold else 0.05

        model = SINDy(poly_degree=poly_degree, threshold=threshold_value)
        model.fit(signal.reshape(-1, 1), dt=self._time_step())
        equations = model.equations()
        normalized = [
            eq if eq.startswith("dx") else f"dx{i}/dt = {eq}"
            for i, eq in enumerate(equations)
        ]
        return (
            "SINDy: "
            f"degree={poly_degree}, threshold={threshold_value:.4g}, "
            f"equations={'; '.join(normalized)}"
        )

    def _run_wavelet(self) -> str:
        """대표 신호로 Wavelet/STFT quick diagnostic을 실행한다."""
        import numpy as np

        from naviertwin.core.flow_analysis.statistics.wavelet import stft_fallback

        signal = self._representative_signal()
        if signal.size < 4:
            return "Wavelet/STFT: 최소 4개 이상의 샘플이 필요합니다."

        n_window = max(4, min(64, signal.size))
        result = stft_fallback(signal, dt=self._time_step(), n_window=n_window)
        spec = np.asarray(result["spectrogram"], dtype=float)
        return (
            f"STFT: spectrogram={spec.shape}, "
            f"peak_power={float(spec.max() if spec.size else 0.0):.4g}"
        )

    def _run_boundary_layer(self) -> str:
        """y 좌표와 첫 번째 속도/스칼라 필드로 경계층 두께를 계산한다."""
        import numpy as np

        from naviertwin.core.flow_analysis.boundary_layer.boundary_layer import (
            boundary_layer_thicknesses,
        )

        mesh = self._dataset.mesh  # type: ignore[union-attr]
        if not hasattr(mesh, "points") or len(mesh.points) < 3:
            return "Boundary Layer: y 프로파일을 만들 수 있는 메쉬 좌표가 필요합니다."

        y = np.asarray(mesh.points, dtype=float)[:, 1]
        values = self._first_field_values()
        if values.ndim > 1:
            values = np.linalg.norm(values, axis=-1)
        y_profile, u_profile = self._mean_profile(y, np.asarray(values, dtype=float).ravel())
        if y_profile.size < 3:
            return "Boundary Layer: 최소 3개 이상의 y 위치가 필요합니다."

        u_profile = np.abs(u_profile)
        u_inf = float(np.max(u_profile))
        if u_inf <= 0:
            return "Boundary Layer: 양의 자유류 속도 추정값이 필요합니다."

        out = boundary_layer_thicknesses(y_profile, u_profile, U_inf=u_inf)
        return (
            "Boundary Layer: "
            f"δ99={out['delta99']:.4g}, δ*={out['delta_star']:.4g}, "
            f"θ={out['theta']:.4g}, H={out['H']:.4g}"
        )

    def _run_nondim(self) -> str:
        """기본 공기 물성으로 주요 무차원수를 계산한다."""
        from naviertwin.core.flow_analysis.thermofluids.nondim import (
            nusselt,
            peclet,
            prandtl,
            reynolds,
        )

        re = reynolds(rho=1.225, U=10.0, L=1.0, mu=1.8e-5)
        pr = prandtl(mu=1.8e-5, cp=1005.0, k=0.026)
        nu = nusselt(h=50.0, L=1.0, k=0.026)
        pe = peclet(re, pr)
        return f"Nondim: Re={re:.4g}, Pr={pr:.4g}, Nu={nu:.4g}, Pe={pe:.4g}"

    def _run_ftle(self) -> str:
        """내장 2D 비정상 유동으로 FTLE/LCS quick diagnostic을 실행한다."""
        import numpy as np

        from naviertwin.core.flow_analysis.vortex.lcs import compute_ftle_2d

        def u_fn(t: float, x: object, y: object) -> object:
            return np.sin(np.pi * x) * np.cos(np.pi * y + 0.2 * t)

        def v_fn(t: float, x: object, y: object) -> object:
            return -np.cos(np.pi * x + 0.2 * t) * np.sin(np.pi * y)

        ftle = compute_ftle_2d(u_fn, v_fn, nx=12, ny=12, T=0.5, dt=0.1)
        return f"FTLE: grid={ftle.shape}, range=[{ftle.min():.4g}, {ftle.max():.4g}]"

    def _run_pgd(self) -> str:
        """대표 3D 텐서로 PGD quick decomposition을 실행한다."""
        import numpy as np

        from naviertwin.core.flow_analysis.modal.pgd import compute_pgd_3d, reconstruct_pgd

        signal = self._representative_signal()
        if signal.size >= 64:
            tensor = signal[:64].reshape(4, 4, 4)
        else:
            x = np.linspace(0.0, 1.0, 4)
            y = np.linspace(0.0, 1.0, 4)
            z = np.linspace(0.0, 1.0, 4)
            tensor = np.sin(np.pi * x)[:, None, None] * np.cos(np.pi * y)[None, :, None]
            tensor = tensor * (1.0 + z[None, None, :])
        modes = compute_pgd_3d(tensor, n_modes=3, max_iter=50)
        rec = reconstruct_pgd(modes, tensor.shape)
        rel = float(np.linalg.norm(tensor - rec) / (np.linalg.norm(tensor) + 1e-30))
        return f"PGD: modes={len(modes)}, relative_residual={rel:.4g}"

    def _run_entropy_generation(self) -> str:
        """2D 열유동 샘플로 엔트로피 생성률 quick diagnostic을 실행한다."""
        import numpy as np

        from naviertwin.core.flow_analysis.thermofluids.entropy_gen import (
            entropy_generation_2d,
        )

        x = np.linspace(0.0, 1.0, 12)
        y = np.linspace(0.0, 1.0, 12)
        X, Y = np.meshgrid(x, y)
        u = X * (1.0 - Y)
        v = 0.1 * Y
        temp = 300.0 + 5.0 * X + 2.0 * Y
        s_gen = entropy_generation_2d(
            u, v, temp, dx=x[1] - x[0], dy=y[1] - y[0], mu=1e-3, k=0.026
        )
        return f"Entropy Generation: mean={s_gen.mean():.4g}, max={s_gen.max():.4g}"

    def _field_snapshots(self) -> object:
        """첫 번째 필드의 스냅샷 행렬을 반환한다."""
        import numpy as np

        if self._dataset is None or not self._dataset.field_names:
            raise RuntimeError("분석할 필드가 없습니다.")
        field = self._dataset.field_names[0]
        return np.asarray(self._dataset.extract_field_snapshots(field), dtype=float)

    def _first_field_values(self) -> object:
        """첫 번째 필드의 현재 메쉬 값을 반환한다."""
        import numpy as np

        if self._dataset is None or not self._dataset.field_names:
            raise RuntimeError("분석할 필드가 없습니다.")
        mesh = self._dataset.mesh
        field = self._dataset.field_names[0]
        if field in mesh.point_data:
            return np.asarray(mesh.point_data[field], dtype=float)
        if field in mesh.cell_data:
            return np.asarray(mesh.cell_data[field], dtype=float)
        raise RuntimeError(f"필드 '{field}'가 메쉬에 없습니다.")

    def _representative_signal(self) -> object:
        """시계열 또는 필드 벡터에서 1D 대표 신호를 구성한다."""
        import numpy as np

        snapshots = self._field_snapshots()
        if snapshots.shape[1] > 1:
            return snapshots.mean(axis=0).astype(float)
        values = self._first_field_values()
        if values.ndim > 1:
            values = np.linalg.norm(values, axis=-1)
        return np.asarray(values, dtype=float).ravel()

    def _time_step(self) -> float:
        """dataset time_steps에서 대표 시간 간격을 추정한다."""
        import numpy as np

        if self._dataset is None or len(self._dataset.time_steps) < 2:
            return 1.0
        diffs = np.diff(np.asarray(self._dataset.time_steps, dtype=float))
        positive = diffs[diffs > 0]
        return float(positive.mean()) if positive.size else 1.0

    @staticmethod
    def _mean_profile(y: object, values: object) -> tuple[object, object]:
        """같은 y 좌표의 값을 평균해 단조 프로파일을 만든다."""
        import numpy as np

        y_arr = np.asarray(y, dtype=float)
        v_arr = np.asarray(values, dtype=float)
        order = np.argsort(y_arr)
        y_sorted = y_arr[order]
        v_sorted = v_arr[order]
        unique_y, inverse = np.unique(y_sorted, return_inverse=True)
        sums = np.zeros_like(unique_y, dtype=float)
        counts = np.zeros_like(unique_y, dtype=float)
        np.add.at(sums, inverse, v_sorted)
        np.add.at(counts, inverse, 1.0)
        return unique_y, sums / np.maximum(counts, 1.0)

    @staticmethod
    def _method_index(method: str) -> int:
        for idx, (_, key) in enumerate(_ANALYSIS_METHODS):
            if key == method:
                return idx
        raise ValueError(f"unknown analysis method: {method}")
