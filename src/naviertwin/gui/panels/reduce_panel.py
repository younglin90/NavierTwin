"""차원 축소 패널 — POD / Randomized POD / DMD 실행 및 에너지 누적 곡선.

Signals:
    reduction_done(str, object): 축소 완료 시 (방법 이름, reducer 객체) 발생.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from naviertwin.core.cfd_reader.base import CFDDataset


class ReducePanel(QWidget):
    """차원 축소 탭 패널.

    Signals:
        reduction_done: 축소 완료 시 reducer 객체와 함께 발생.
    """

    reduction_done = Signal(str, object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._dataset: Optional[CFDDataset] = None
        self._reducer: Optional[object] = None
        self._setup_ui()

    # ──────────────────────────────────────────────────────────────────
    # UI 초기화
    # ──────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 좌측 컨트롤
        left = QWidget()
        left.setFixedWidth(280)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        title = QLabel("Dimensionality Reduction")
        title.setObjectName("titleLabel")
        left_layout.addWidget(title)

        # 방법 선택
        method_group = QGroupBox("축소 방법")
        method_layout = QFormLayout(method_group)

        self._method_combo = QComboBox()
        self._method_combo.addItems(["Snapshot POD", "Randomized POD", "DMD"])
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_layout.addRow("방법:", self._method_combo)

        self._field_combo = QComboBox()
        method_layout.addRow("필드:", self._field_combo)

        left_layout.addWidget(method_group)

        # 파라미터 스택
        param_group = QGroupBox("파라미터")
        param_layout = QVBoxLayout(param_group)
        self._param_stack = QStackedWidget()
        self._param_stack.addWidget(self._build_pod_params())       # POD
        self._param_stack.addWidget(self._build_rand_pod_params())  # Randomized POD
        self._param_stack.addWidget(self._build_dmd_params())       # DMD
        param_layout.addWidget(self._param_stack)
        left_layout.addWidget(param_group)

        # 에너지 임계값
        energy_group = QGroupBox("에너지 임계값")
        energy_layout = QFormLayout(energy_group)
        self._energy_spin = QDoubleSpinBox()
        self._energy_spin.setRange(0.5, 1.0)
        self._energy_spin.setValue(0.99)
        self._energy_spin.setDecimals(3)
        self._energy_spin.setSingleStep(0.01)
        energy_layout.addRow("누적 에너지 ≥:", self._energy_spin)
        left_layout.addWidget(energy_group)

        # 실행 버튼
        self._run_btn = QPushButton("축소 실행")
        self._run_btn.setObjectName("primaryButton")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run_reduction)
        left_layout.addWidget(self._run_btn)

        left_layout.addStretch()
        layout.addWidget(left)

        # 우측: 결과
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 에너지 곡선 플롯 자리 (텍스트 대체)
        energy_plot_group = QGroupBox("에너지 누적 곡선")
        energy_plot_layout = QVBoxLayout(energy_plot_group)
        self._energy_plot_label = QLabel("축소 완료 후 표시됩니다.")
        self._energy_plot_label.setAlignment(
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.AlignmentFlag.AlignCenter
        )
        self._energy_plot_label.setStyleSheet(
            "border: 1px dashed #4A4A70; padding: 20px; color: #9090B0;"
        )
        self._energy_plot_label.setMinimumHeight(180)
        energy_plot_layout.addWidget(self._energy_plot_label)
        right_layout.addWidget(energy_plot_group)

        # 결과 로그
        log_group = QGroupBox("결과 / 로그")
        log_layout = QVBoxLayout(log_group)
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        log_layout.addWidget(self._result_text)
        right_layout.addWidget(log_group)

        layout.addWidget(right, stretch=1)

    def _build_pod_params(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(4, 4, 4, 4)
        n_modes = QSpinBox()
        n_modes.setRange(1, 9999)
        n_modes.setValue(20)
        n_modes.setObjectName("n_modes")
        form.addRow("모드 수:", n_modes)
        center = QComboBox()
        center.addItems(["True", "False"])
        form.addRow("Mean centering:", center)
        return w

    def _build_rand_pod_params(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(4, 4, 4, 4)
        n_modes = QSpinBox()
        n_modes.setRange(1, 9999)
        n_modes.setValue(20)
        n_modes.setObjectName("n_modes")
        form.addRow("모드 수:", n_modes)
        n_iter = QSpinBox()
        n_iter.setRange(1, 20)
        n_iter.setValue(4)
        form.addRow("Power iterations:", n_iter)
        return w

    def _build_dmd_params(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(4, 4, 4, 4)
        method = QComboBox()
        method.addItems(["fbdmd", "dmd", "hodmd", "spdmd"])
        form.addRow("DMD 방법:", method)
        n_modes = QSpinBox()
        n_modes.setRange(1, 9999)
        n_modes.setValue(20)
        form.addRow("모드 수:", n_modes)
        dt = QDoubleSpinBox()
        dt.setRange(1e-6, 1e3)
        dt.setValue(0.01)
        dt.setDecimals(6)
        form.addRow("Δt (s):", dt)
        return w

    # ──────────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────────

    def set_dataset(self, dataset: CFDDataset) -> None:
        """분석할 CFDDataset을 설정한다."""
        self._dataset = dataset
        self._run_btn.setEnabled(True)
        self._field_combo.clear()
        self._field_combo.addItems(dataset.field_names)

    # ──────────────────────────────────────────────────────────────────
    # 슬롯
    # ──────────────────────────────────────────────────────────────────

    def _on_method_changed(self, idx: int) -> None:
        self._param_stack.setCurrentIndex(idx)

    def _run_reduction(self) -> None:
        if self._dataset is None:
            return
        method_idx = self._method_combo.currentIndex()
        field = self._field_combo.currentText()

        try:
            snapshots = self._extract_snapshots(field)
            if method_idx == 0:
                self._run_pod(snapshots)
            elif method_idx == 1:
                self._run_rand_pod(snapshots)
            else:
                self._run_dmd(snapshots)
        except Exception as exc:
            self._result_text.append(f"[ERROR] {exc}\n")

    def _extract_snapshots(self, field: str) -> np.ndarray:
        """메쉬에서 스냅샷 행렬을 추출한다 (n_features, 1)."""
        mesh = self._dataset.mesh  # type: ignore[union-attr]
        if field in mesh.point_data:
            arr = np.array(mesh.point_data[field], dtype=float)
        elif field in mesh.cell_data:
            arr = np.array(mesh.cell_data[field], dtype=float)
        else:
            raise ValueError(f"필드 '{field}'가 메쉬에 없습니다.")

        if arr.ndim > 1:
            # 벡터 → 크기
            arr = np.linalg.norm(arr, axis=1)
        return arr.reshape(-1, 1)  # (n_features, 1)

    def _run_pod(self, snapshots: np.ndarray) -> None:
        from naviertwin.core.dimensionality_reduction.linear.pod import SnapshotPOD

        page = self._param_stack.widget(0)
        n_modes_spin = page.findChild(QSpinBox, "n_modes")
        n_modes = n_modes_spin.value() if n_modes_spin else 20
        n_modes = min(n_modes, snapshots.shape[1])

        reducer = SnapshotPOD(n_modes=n_modes)
        reducer.fit(snapshots)
        energy = reducer.energy_ratio

        self._reducer = reducer
        msg = (
            f"Snapshot POD 완료\n"
            f"  modes: {n_modes}, energy: {energy[-1]*100:.2f}%\n"
            f"  singular values: {reducer.singular_values[:5]}\n"
        )
        self._result_text.append(msg)
        self._update_energy_plot(energy)
        self.reduction_done.emit("pod", reducer)

    def _run_rand_pod(self, snapshots: np.ndarray) -> None:
        from naviertwin.core.dimensionality_reduction.linear.randomized_svd import RandomizedPOD

        page = self._param_stack.widget(1)
        n_modes_spin = page.findChild(QSpinBox, "n_modes")
        n_modes = n_modes_spin.value() if n_modes_spin else 20
        n_modes = min(n_modes, min(snapshots.shape))

        reducer = RandomizedPOD(n_modes=n_modes)
        reducer.fit(snapshots)
        energy = reducer.energy_ratio

        self._reducer = reducer
        msg = (
            f"Randomized POD 완료\n"
            f"  modes: {n_modes}, energy: {energy[-1]*100:.2f}%\n"
        )
        self._result_text.append(msg)
        self._update_energy_plot(energy)
        self.reduction_done.emit("randomized_pod", reducer)

    def _run_dmd(self, snapshots: np.ndarray) -> None:
        from naviertwin.core.flow_analysis.modal.dmd import DMDAnalyzer

        page = self._param_stack.widget(2)
        combos = page.findChildren(QComboBox)
        spins = page.findChildren(QDoubleSpinBox)
        method = combos[0].currentText() if combos else "fbdmd"
        dt = spins[0].value() if spins else 0.01

        # DMD는 최소 2개 스냅샷 필요 → 단일 스냅샷이면 복제
        if snapshots.shape[1] < 2:
            snapshots = np.hstack([snapshots, snapshots + 1e-10])

        analyzer = DMDAnalyzer(method=method)
        analyzer.fit(snapshots, dt=dt)

        freqs = analyzer.frequencies
        n_modes = len(freqs) if freqs is not None else 0
        msg = f"DMD ({method}) 완료\n  modes: {n_modes}\n"
        if freqs is not None and len(freqs) > 0:
            msg += f"  top freq: {sorted(abs(freqs))[:3]}\n"
        self._result_text.append(msg)
        self.reduction_done.emit("dmd", analyzer)

    def _update_energy_plot(self, energy: np.ndarray) -> None:
        """에너지 누적 곡선을 텍스트로 표시한다."""
        lines = []
        for i, e in enumerate(energy[:10], 1):
            bar = "█" * int(e * 20)
            lines.append(f"Mode {i:2d}: {bar:<20} {e*100:.1f}%")
        self._energy_plot_label.setText("\n".join(lines))
        self._energy_plot_label.setAlignment(
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.AlignmentFlag.AlignLeft
            | __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.AlignmentFlag.AlignTop
        )
        self._energy_plot_label.setStyleSheet(
            "font-family: monospace; padding: 10px; color: #CBA6F7;"
        )
