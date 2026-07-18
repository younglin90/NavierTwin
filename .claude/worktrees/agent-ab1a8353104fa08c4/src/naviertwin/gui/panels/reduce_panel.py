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
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from naviertwin.core.cfd_reader.base import CFDDataset


def _format_energy_line(entry: tuple[int, float]) -> str:
    index, energy = entry
    bar = "█" * int(energy * 20)
    return f"Mode {index:2d}: {bar:<20} {energy * 100:.1f}%"


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
        self._active_field: str = ""
        self._reduction_artifact: dict[str, object] | None = None
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
        self._method_combo.addItems(
            [
                "Snapshot POD",
                "Randomized POD",
                "Incremental POD",
                "MRPOD",
                "DMD",
                "Autoencoder",
                "VAE",
            ]
        )
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_layout.addRow("방법:", self._method_combo)

        self._field_combo = QComboBox()
        method_layout.addRow("필드:", self._field_combo)

        left_layout.addWidget(method_group)

        # 파라미터 스택
        param_group = QGroupBox("파라미터")
        param_layout = QVBoxLayout(param_group)
        self._param_stack = QStackedWidget()
        self._param_stack.addWidget(self._build_pod_params())              # POD
        self._param_stack.addWidget(self._build_rand_pod_params())         # Randomized POD
        self._param_stack.addWidget(self._build_incremental_pod_params())  # Incremental POD
        self._param_stack.addWidget(self._build_mrpod_params())            # MRPOD
        self._param_stack.addWidget(self._build_dmd_params())              # DMD
        self._param_stack.addWidget(self._build_autoencoder_params())      # Autoencoder
        self._param_stack.addWidget(self._build_vae_params())              # VAE
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
        # 검증된 변형 우선 — fbdmd 는 이상적인 데이터에서도 발산 사례가 있고
        # hodmd 는 지연 임베딩으로 reconstruct 와 차원이 안 맞는다 (dmd.py Note).
        method.addItems(["dmd", "spdmd", "fbdmd", "hodmd"])
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

    def _build_incremental_pod_params(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(4, 4, 4, 4)
        n_modes = QSpinBox()
        n_modes.setRange(1, 9999)
        n_modes.setValue(20)
        n_modes.setObjectName("n_modes")
        form.addRow("모드 수:", n_modes)
        forget = QDoubleSpinBox()
        forget.setRange(0.1, 1.0)
        forget.setSingleStep(0.05)
        forget.setValue(1.0)
        forget.setDecimals(2)
        forget.setObjectName("forget_factor")
        form.addRow("Forget factor:", forget)
        return w

    def _build_mrpod_params(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(4, 4, 4, 4)
        n_scales = QSpinBox()
        n_scales.setRange(1, 10)
        n_scales.setValue(3)
        n_scales.setObjectName("n_scales")
        form.addRow("스케일 수:", n_scales)
        n_modes = QSpinBox()
        n_modes.setRange(1, 9999)
        n_modes.setValue(10)
        n_modes.setObjectName("n_modes_per_scale")
        form.addRow("스케일당 모드:", n_modes)
        return w

    def _build_autoencoder_params(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(4, 4, 4, 4)
        latent = QSpinBox()
        latent.setRange(1, 256)
        latent.setValue(4)
        latent.setObjectName("latent_dim")
        form.addRow("Latent dim:", latent)
        hidden = QSpinBox()
        hidden.setRange(4, 4096)
        hidden.setValue(32)
        hidden.setObjectName("hidden_dim")
        form.addRow("Hidden dim:", hidden)
        epochs = QSpinBox()
        epochs.setRange(1, 2000)
        epochs.setValue(20)
        epochs.setObjectName("max_epochs")
        form.addRow("Epoch:", epochs)
        return w

    def _build_vae_params(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(4, 4, 4, 4)
        latent = QSpinBox()
        latent.setRange(1, 256)
        latent.setValue(4)
        latent.setObjectName("latent_dim")
        form.addRow("Latent dim:", latent)
        hidden = QSpinBox()
        hidden.setRange(4, 4096)
        hidden.setValue(32)
        hidden.setObjectName("hidden_dim")
        form.addRow("Hidden dim:", hidden)
        epochs = QSpinBox()
        epochs.setRange(1, 2000)
        epochs.setValue(20)
        epochs.setObjectName("max_epochs")
        form.addRow("Epoch:", epochs)
        beta = QDoubleSpinBox()
        beta.setRange(0.0, 100.0)
        beta.setValue(1.0)
        beta.setDecimals(3)
        beta.setObjectName("beta")
        form.addRow("β:", beta)
        return w

    # ──────────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────────

    def set_dataset(self, dataset: CFDDataset) -> None:
        """분석할 CFDDataset을 설정한다."""
        self._dataset = dataset
        self._reduction_artifact = None
        self._run_btn.setEnabled(True)
        self._field_combo.clear()
        self._field_combo.addItems(dataset.field_names)

    def get_reduction_artifact(self) -> dict[str, object] | None:
        """마지막 축소 결과 아티팩트를 반환한다."""
        return self._reduction_artifact

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
        self._active_field = field

        try:
            snapshots = self._extract_snapshots(field)
            if method_idx == 0:
                self._run_pod(snapshots)
            elif method_idx == 1:
                self._run_rand_pod(snapshots)
            elif method_idx == 2:
                self._run_incremental_pod(snapshots)
            elif method_idx == 3:
                self._run_mrpod(snapshots)
            elif method_idx == 4:
                self._run_dmd(snapshots)
            elif method_idx == 5:
                self._run_autoencoder(snapshots)
            else:
                self._run_vae(snapshots)
        except Exception as exc:
            self._result_text.append(f"[ERROR] {exc}\n")

    def _extract_snapshots(self, field: str) -> np.ndarray:
        """메쉬에서 스냅샷 행렬을 추출한다.

        다중 타임스텝 데이터를 찾을 수 있으면 (n_features, n_steps) 형태로
        반환하고, 그렇지 않으면 기존과 동일하게 단일 스냅샷
        (n_features, 1)을 반환한다.
        """
        if self._dataset is None:
            raise RuntimeError("dataset이 설정되지 않았습니다.")
        return self._dataset.extract_field_snapshots(field)

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
        reducer.training_metadata = {
            "dataset_id": id(self._dataset) if self._dataset is not None else None,
            "field_name": self._active_field,
            "n_modes": int(getattr(reducer, "n_components", n_modes)),
            "method": "pod",
        }
        coeffs = reducer.encode(snapshots)
        params = (
            np.asarray(self._dataset.time_steps[: coeffs.shape[0]], dtype=float)
            if self._dataset is not None and len(self._dataset.time_steps) >= coeffs.shape[0]
            else np.arange(coeffs.shape[0], dtype=float)
        )
        self._reduction_artifact = {
            "method": "pod",
            "field_name": self._active_field,
            "snapshots": snapshots,
            "coeffs": coeffs,
            "params": params.reshape(-1, 1),
            "dataset_id": id(self._dataset) if self._dataset is not None else None,
        }
        msg = (
            f"Snapshot POD 완료\n"
            f"  modes: {n_modes}, energy: {energy[-1]*100:.2f}%\n"
            f"  singular values: {self._singular_values_preview(reducer)}\n"
        )
        self._result_text.append(msg)
        self._update_energy_plot(energy)
        self.reduction_done.emit("pod", reducer)

    @staticmethod
    def _singular_values_preview(reducer: object, limit: int = 5) -> np.ndarray:
        """Reducer 구현별 특이값 속성명을 흡수해 로그용 preview를 반환한다."""
        values = getattr(reducer, "singular_values_", None)
        if values is None:
            values = getattr(reducer, "singular_values", None)
        if values is None:
            return np.array([], dtype=float)
        return np.asarray(values, dtype=float)[:limit]

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
        reducer.training_metadata = {
            "dataset_id": id(self._dataset) if self._dataset is not None else None,
            "field_name": self._active_field,
            "n_modes": int(getattr(reducer, "n_components", n_modes)),
            "method": "randomized_pod",
        }
        coeffs = reducer.encode(snapshots)
        params = (
            np.asarray(self._dataset.time_steps[: coeffs.shape[0]], dtype=float)
            if self._dataset is not None and len(self._dataset.time_steps) >= coeffs.shape[0]
            else np.arange(coeffs.shape[0], dtype=float)
        )
        self._reduction_artifact = {
            "method": "randomized_pod",
            "field_name": self._active_field,
            "snapshots": snapshots,
            "coeffs": coeffs,
            "params": params.reshape(-1, 1),
            "dataset_id": id(self._dataset) if self._dataset is not None else None,
        }
        msg = (
            f"Randomized POD 완료\n"
            f"  modes: {n_modes}, energy: {energy[-1]*100:.2f}%\n"
        )
        self._result_text.append(msg)
        self._update_energy_plot(energy)
        self.reduction_done.emit("randomized_pod", reducer)

    def _run_dmd(self, snapshots: np.ndarray) -> None:
        from naviertwin.core.flow_analysis.modal.dmd import DMDAnalyzer

        page = self._param_stack.widget(4)
        combos = page.findChildren(QComboBox)
        spins = page.findChildren(QDoubleSpinBox)
        method = combos[0].currentText() if combos else "dmd"
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
        analyzer.training_metadata = {
            "dataset_id": id(self._dataset) if self._dataset is not None else None,
            "field_name": self._active_field,
            "n_modes": int(n_modes),
            "method": f"dmd:{method}",
        }
        self._reduction_artifact = {
            "method": f"dmd:{method}",
            "field_name": self._active_field,
            "snapshots": snapshots,
            "coeffs": None,
            "params": None,
            "dataset_id": id(self._dataset) if self._dataset is not None else None,
        }
        self.reduction_done.emit("dmd", analyzer)

    def _run_incremental_pod(self, snapshots: np.ndarray) -> None:
        from naviertwin.core.dimensionality_reduction.linear.incremental_pod import (
            IncrementalPOD,
        )

        page = self._param_stack.widget(2)
        n_modes_spin = page.findChild(QSpinBox, "n_modes")
        forget_spin = page.findChild(QDoubleSpinBox, "forget_factor")
        n_modes = n_modes_spin.value() if n_modes_spin else 20
        forget_factor = forget_spin.value() if forget_spin else 1.0

        reducer = IncrementalPOD(n_modes=min(n_modes, snapshots.shape[1]), forget_factor=forget_factor)
        reducer.fit(snapshots)
        energy = reducer.energy_ratio

        self._reducer = reducer
        reducer.training_metadata = {
            "dataset_id": id(self._dataset) if self._dataset is not None else None,
            "field_name": self._active_field,
            "n_modes": int(getattr(reducer, "n_components", n_modes)),
            "method": "incremental_pod",
        }
        coeffs = reducer.encode(snapshots)
        params = (
            np.asarray(self._dataset.time_steps[: coeffs.shape[0]], dtype=float)
            if self._dataset is not None and len(self._dataset.time_steps) >= coeffs.shape[0]
            else np.arange(coeffs.shape[0], dtype=float)
        )
        self._reduction_artifact = {
            "method": "incremental_pod",
            "field_name": self._active_field,
            "snapshots": snapshots,
            "coeffs": coeffs,
            "params": params.reshape(-1, 1),
            "dataset_id": id(self._dataset) if self._dataset is not None else None,
        }
        msg = (
            f"Incremental POD 완료\n"
            f"  modes: {reducer.n_components}, forget_factor: {forget_factor}\n"
        )
        self._result_text.append(msg)
        self._update_energy_plot(np.asarray(energy, dtype=float))
        self.reduction_done.emit("incremental_pod", reducer)

    def _run_mrpod(self, snapshots: np.ndarray) -> None:
        from naviertwin.core.dimensionality_reduction.linear.mrpod import MRPOD

        page = self._param_stack.widget(3)
        n_scales_spin = page.findChild(QSpinBox, "n_scales")
        n_modes_spin = page.findChild(QSpinBox, "n_modes_per_scale")
        n_scales = n_scales_spin.value() if n_scales_spin else 3
        n_modes = n_modes_spin.value() if n_modes_spin else 10

        reducer = MRPOD(n_scales=n_scales, n_modes_per_scale=min(n_modes, snapshots.shape[1]))
        reducer.fit(snapshots)
        energy = reducer.get_energy_fraction()

        self._reducer = reducer
        reducer.training_metadata = {
            "dataset_id": id(self._dataset) if self._dataset is not None else None,
            "field_name": self._active_field,
            "n_modes": int(getattr(reducer, "n_components", n_scales * n_modes)),
            "method": "mrpod",
        }
        coeffs = reducer.encode(snapshots)
        params = (
            np.asarray(self._dataset.time_steps[: coeffs.shape[0]], dtype=float)
            if self._dataset is not None and len(self._dataset.time_steps) >= coeffs.shape[0]
            else np.arange(coeffs.shape[0], dtype=float)
        )
        self._reduction_artifact = {
            "method": "mrpod",
            "field_name": self._active_field,
            "snapshots": snapshots,
            "coeffs": coeffs,
            "params": params.reshape(-1, 1),
            "dataset_id": id(self._dataset) if self._dataset is not None else None,
        }
        msg = (
            f"MRPOD 완료\n"
            f"  scales: {n_scales}, modes/scale: {n_modes}, total_modes: {reducer.n_components}\n"
        )
        self._result_text.append(msg)
        self._update_energy_plot(np.asarray(energy, dtype=float))
        self.reduction_done.emit("mrpod", reducer)

    def _run_autoencoder(self, snapshots: np.ndarray) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.autoencoder import (
            Autoencoder,
        )

        page = self._param_stack.widget(5)
        latent = self._spin_value(page, "latent_dim", 4)
        hidden = self._spin_value(page, "hidden_dim", 32)
        epochs = self._spin_value(page, "max_epochs", 20)
        latent = min(latent, max(1, min(snapshots.shape)))

        reducer = Autoencoder(
            latent_dim=latent,
            hidden_dims=[hidden],
            max_epochs=epochs,
        )
        reducer.fit(snapshots)
        self._finish_nonlinear_reduction(
            method="autoencoder",
            reducer=reducer,
            snapshots=snapshots,
            extra=f"hidden={hidden}, epochs={epochs}",
        )

    def _run_vae(self, snapshots: np.ndarray) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.vae import VAE

        page = self._param_stack.widget(6)
        latent = self._spin_value(page, "latent_dim", 4)
        hidden = self._spin_value(page, "hidden_dim", 32)
        epochs = self._spin_value(page, "max_epochs", 20)
        beta_widget = page.findChild(QDoubleSpinBox, "beta")
        beta = beta_widget.value() if beta_widget else 1.0
        latent = min(latent, max(1, min(snapshots.shape)))

        reducer = VAE(
            latent_dim=latent,
            hidden_dims=[hidden],
            max_epochs=epochs,
            beta=beta,
        )
        reducer.fit(snapshots)
        self._finish_nonlinear_reduction(
            method="vae",
            reducer=reducer,
            snapshots=snapshots,
            extra=f"hidden={hidden}, epochs={epochs}, beta={beta:g}",
        )

    def _finish_nonlinear_reduction(
        self,
        *,
        method: str,
        reducer: object,
        snapshots: np.ndarray,
        extra: str,
    ) -> None:
        self._reducer = reducer
        n_modes = int(getattr(reducer, "n_components", 0))
        setattr(
            reducer,
            "training_metadata",
            {
                "dataset_id": id(self._dataset) if self._dataset is not None else None,
                "field_name": self._active_field,
                "n_modes": n_modes,
                "method": method,
            },
        )
        coeffs = reducer.encode(snapshots)  # type: ignore[attr-defined]
        recon = reducer.decode(coeffs)  # type: ignore[attr-defined]
        rmse = float(np.sqrt(np.mean((np.asarray(recon) - snapshots) ** 2)))
        params = (
            np.asarray(self._dataset.time_steps[: coeffs.shape[0]], dtype=float)
            if self._dataset is not None and len(self._dataset.time_steps) >= coeffs.shape[0]
            else np.arange(coeffs.shape[0], dtype=float)
        )
        self._reduction_artifact = {
            "method": method,
            "field_name": self._active_field,
            "snapshots": snapshots,
            "coeffs": coeffs,
            "params": params.reshape(-1, 1),
            "dataset_id": id(self._dataset) if self._dataset is not None else None,
        }
        losses = getattr(reducer, "train_losses_", [])
        loss_tail = float(losses[-1]) if losses else float("nan")
        self._result_text.append(
            f"{method.upper()} 완료\n"
            f"  latent: {n_modes}, recon_rmse: {rmse:.6g}, final_loss: {loss_tail:.6g}\n"
            f"  {extra}\n"
        )
        self._update_energy_plot(self._loss_to_progress(losses))
        self.reduction_done.emit(method, reducer)

    @staticmethod
    def _spin_value(page: QWidget, object_name: str, default: int) -> int:
        widget = page.findChild(QSpinBox, object_name)
        return int(widget.value()) if widget is not None else default

    @staticmethod
    def _loss_to_progress(losses: object) -> np.ndarray:
        values = np.asarray(losses, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.array([1.0], dtype=float)
        first = max(float(values[0]), 1e-12)
        progress = 1.0 - np.clip(values[:10] / first, 0.0, 1.0)
        if progress.size > 0:
            progress[-1] = max(progress[-1], 1e-6)
        return progress

    def _update_energy_plot(self, energy: np.ndarray) -> None:
        """에너지 누적 곡선을 텍스트로 표시한다."""
        entries = enumerate(energy[:10], 1)
        self._energy_plot_label.setText("\n".join(map(_format_energy_line, entries)))
        self._energy_plot_label.setAlignment(
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.AlignmentFlag.AlignLeft
            | __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.AlignmentFlag.AlignTop
        )
        self._energy_plot_label.setStyleSheet(
            "font-family: monospace; padding: 10px; color: #CBA6F7;"
        )
