"""유동 분석 패널 — Q-criterion, FFT/PSD, y+ 계산 및 결과 시각화.

Signals:
    analysis_done(str, object): 분석 완료 시 (분석 이름, 결과 메쉬) 발생.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from naviertwin.core.cfd_reader.base import CFDDataset


class AnalyzePanel(QWidget):
    """유동 분석 탭 패널.

    Q-criterion / λ₂, FFT/PSD, y+ 계산 기능을 제공한다.

    Signals:
        analysis_done: 분석 완료 시 결과 메쉬와 함께 발생.
    """

    analysis_done = Signal(str, object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._dataset: Optional[CFDDataset] = None
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
        self._method_list.addItems([
            "Q-criterion",
            "λ₂ Criterion",
            "FFT / PSD",
            "y+ (Wall Units)",
        ])
        self._method_list.currentRowChanged.connect(self._on_method_selected)
        method_layout.addWidget(self._method_list)
        left_layout.addWidget(method_group)

        # 파라미터 스택
        param_group = QGroupBox("파라미터")
        param_layout = QVBoxLayout(param_group)
        self._param_stack = QStackedWidget()

        # Q-criterion 파라미터
        self._param_stack.addWidget(self._build_qcrit_params())
        # λ₂ 파라미터 (동일)
        self._param_stack.addWidget(self._build_qcrit_params())
        # FFT 파라미터
        self._param_stack.addWidget(self._build_fft_params())
        # y+ 파라미터
        self._param_stack.addWidget(self._build_yplus_params())

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

        # 우측: 결과
        right_splitter = QSplitter()
        right_splitter.setOrientation(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.Orientation.Vertical)

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

        right_splitter.setSizes([400, 40])
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

    def _build_yplus_params(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(4, 4, 4, 4)

        wss_combo = QComboBox()
        wss_combo.addItems(["wallShearStress", "tau_w"])
        form.addRow("벽면 전단응력 필드:", wss_combo)

        rho_spin = QDoubleSpinBox()
        rho_spin.setRange(0.001, 1e5)
        rho_spin.setValue(1.225)
        rho_spin.setDecimals(4)
        form.addRow("밀도 ρ (kg/m³):", rho_spin)

        nu_spin = QDoubleSpinBox()
        nu_spin.setRange(1e-10, 1.0)
        nu_spin.setValue(1.5e-5)
        nu_spin.setDecimals(8)
        nu_spin.setSingleStep(1e-6)
        form.addRow("동점성계수 ν (m²/s):", nu_spin)

        y_spin = QDoubleSpinBox()
        y_spin.setRange(1e-10, 1.0)
        y_spin.setValue(1e-4)
        y_spin.setDecimals(8)
        y_spin.setSingleStep(1e-5)
        form.addRow("첫 번째 셀 높이 y (m):", y_spin)

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
        # 속도 필드 콤보 업데이트
        for page_idx in [0, 1]:
            page = self._param_stack.widget(page_idx)
            combo = page.findChild(QComboBox, "velocity_combo")
            if combo is not None:
                combo.clear()
                combo.addItems(dataset.field_names)

    # ──────────────────────────────────────────────────────────────────
    # 슬롯
    # ──────────────────────────────────────────────────────────────────

    def _on_method_selected(self, row: int) -> None:
        self._param_stack.setCurrentIndex(row)

    def _run_analysis(self) -> None:
        if self._dataset is None:
            return
        row = self._method_list.currentRow()
        methods = ["q_criterion", "lambda2", "fft_psd", "yplus"]
        method = methods[row] if row >= 0 else "q_criterion"
        try:
            result = self._dispatch(method)
            self._result_text.append(f"[{method}] 완료\n{result}\n")
            self.analysis_done.emit(method, result)
        except Exception as exc:
            self._result_text.append(f"[ERROR] {method}: {exc}\n")

    def _dispatch(self, method: str) -> object:
        from naviertwin.core.flow_analysis.vortex.q_criterion import (
            compute_lambda2,
            compute_q_criterion,
        )

        mesh = self._dataset.mesh  # type: ignore[union-attr]

        if method == "q_criterion":
            page = self._param_stack.widget(0)
            combo = page.findChild(QComboBox, "velocity_combo")
            vel = combo.currentText() if combo else "U"
            result_mesh = compute_q_criterion(mesh, vel)
            vals = result_mesh.point_data.get("Q_criterion")
            if vals is not None:
                return f"Q range: [{vals.min():.4g}, {vals.max():.4g}]"
            return result_mesh

        elif method == "lambda2":
            page = self._param_stack.widget(1)
            combo = page.findChild(QComboBox, "velocity_combo")
            vel = combo.currentText() if combo else "U"
            result_mesh = compute_lambda2(mesh, vel)
            vals = result_mesh.point_data.get("lambda2")
            if vals is not None:
                return f"λ₂ range: [{vals.min():.4g}, {vals.max():.4g}]"
            return result_mesh

        elif method == "fft_psd":
            from naviertwin.core.flow_analysis.statistics.fft_psd import (
                compute_fft,
                find_dominant_frequencies,
            )
            import numpy as np

            page = self._param_stack.widget(2)
            spins = page.findChildren(QDoubleSpinBox)
            dt = spins[0].value() if spins else 0.01
            # 단순히 첫 번째 필드의 첫 번째 포인트 신호를 사용
            field = self._dataset.field_names[0] if self._dataset.field_names else None  # type: ignore[union-attr]
            if field and field in mesh.point_data:
                signal = mesh.point_data[field]
                if signal.ndim > 1:
                    signal = signal[:, 0]
                signal = signal.astype(float)
                freqs, amps = compute_fft(signal, dt)
                peaks = find_dominant_frequencies(freqs, amps, n_peaks=3)
                return f"Top frequencies: {[f'{f:.4g} Hz' for f in peaks]}"
            return "FFT: 필드 없음"

        elif method == "yplus":
            from naviertwin.core.flow_analysis.boundary_layer.yplus import compute_yplus
            import numpy as np

            page = self._param_stack.widget(3)
            spins = page.findChildren(QDoubleSpinBox)
            rho = spins[0].value() if len(spins) > 0 else 1.225
            nu = spins[1].value() if len(spins) > 1 else 1.5e-5
            y_wall = spins[2].value() if len(spins) > 2 else 1e-4

            # 벽면 전단응력 필드 탐색
            wss_combo = page.findChild(QComboBox)
            wss_field = wss_combo.currentText() if wss_combo else "wallShearStress"
            if wss_field in mesh.point_data:
                tau = mesh.point_data[wss_field]
                if tau.ndim > 1:
                    tau = np.linalg.norm(tau, axis=1)
                yplus = compute_yplus(tau.astype(float), rho, nu, y_wall)
                return f"y+ range: [{yplus.min():.4g}, {yplus.max():.4g}], mean={yplus.mean():.4g}"
            return "y+: 벽면 전단응력 필드 없음"

        return "알 수 없는 분석 방법"
