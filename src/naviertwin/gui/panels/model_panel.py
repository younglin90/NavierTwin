"""모델 학습/평가 패널 — Surrogate 모델(RBF/Kriging) 학습 및 검증 지표 표시.

Signals:
    model_trained(str, object): 모델 학습 완료 시 (모델 타입, surrogate) 발생.
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
    QListWidget,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from naviertwin.core.cfd_reader.base import CFDDataset


class ModelPanel(QWidget):
    """Surrogate 모델 학습 패널.

    Signals:
        model_trained: 모델 학습 완료 시 surrogate 객체와 함께 발생.
    """

    model_trained = Signal(str, object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._dataset: Optional[CFDDataset] = None
        self._reducer: Optional[object] = None
        self._surrogate: Optional[object] = None
        self._reduction_artifact: Optional[dict[str, object]] = None
        self._loaded_metadata: dict[str, object] = {}
        self._loss_series: dict[str, list[float]] = {}
        self._setup_ui()

    # ──────────────────────────────────────────────────────────────────
    # UI 초기화
    # ──────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 좌측
        left = QWidget()
        left.setFixedWidth(280)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        title = QLabel("Surrogate Model")
        title.setObjectName("titleLabel")
        left_layout.addWidget(title)

        # 모델 선택
        model_group = QGroupBox("모델 선택")
        model_form = QFormLayout(model_group)

        self._model_combo = QComboBox()
        self._model_combo.addItems(["Kriging (RBF default)", "RBF", "Kriging"])
        model_form.addRow("Surrogate 타입:", self._model_combo)

        left_layout.addWidget(model_group)

        # 학습 데이터 설정
        data_group = QGroupBox("학습 데이터 생성 (데모)")
        data_form = QFormLayout(data_group)

        self._n_samples_spin = QSpinBox()
        self._n_samples_spin.setRange(2, 1000)
        self._n_samples_spin.setValue(20)
        data_form.addRow("샘플 수:", self._n_samples_spin)

        self._n_params_spin = QSpinBox()
        self._n_params_spin.setRange(1, 20)
        self._n_params_spin.setValue(2)
        data_form.addRow("파라미터 차원:", self._n_params_spin)

        self._n_outputs_spin = QSpinBox()
        self._n_outputs_spin.setRange(1, 100)
        self._n_outputs_spin.setValue(5)
        data_form.addRow("출력 차원:", self._n_outputs_spin)

        self._train_ratio_spin = QDoubleSpinBox()
        self._train_ratio_spin.setRange(0.5, 0.95)
        self._train_ratio_spin.setValue(0.8)
        self._train_ratio_spin.setSingleStep(0.05)
        data_form.addRow("학습 비율:", self._train_ratio_spin)

        left_layout.addWidget(data_group)

        self._train_btn = QPushButton("모델 학습")
        self._train_btn.setObjectName("primaryButton")
        self._train_btn.clicked.connect(self._train_model)
        left_layout.addWidget(self._train_btn)

        # ─── 신경 연산자 (v2.0+) ───
        op_group = QGroupBox("신경 연산자 (Operator Learning)")
        op_form = QFormLayout(op_group)

        self._op_combo = QComboBox()
        self._op_combo.addItems(["FNO1D", "FNO2D", "TFNO2D", "DeepONet", "UNet2D", "WNO1D"])
        op_form.addRow("연산자 타입:", self._op_combo)

        self._op_epochs_spin = QSpinBox()
        self._op_epochs_spin.setRange(1, 1000)
        self._op_epochs_spin.setValue(10)
        op_form.addRow("Epoch:", self._op_epochs_spin)

        self._op_samples_spin = QSpinBox()
        self._op_samples_spin.setRange(4, 1000)
        self._op_samples_spin.setValue(20)
        op_form.addRow("데모 샘플 수:", self._op_samples_spin)

        self._op_train_btn = QPushButton("연산자 학습")
        self._op_train_btn.clicked.connect(self._train_operator)
        op_form.addRow(self._op_train_btn)

        left_layout.addWidget(op_group)

        left_layout.addStretch()
        layout.addWidget(left)

        # 우측
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 검증 지표
        metrics_group = QGroupBox("검증 지표")
        metrics_layout = QVBoxLayout(metrics_group)
        self._metrics_list = QListWidget()
        self._metrics_list.setAlternatingRowColors(True)
        self._metrics_list.setMaximumHeight(160)
        metrics_layout.addWidget(self._metrics_list)
        right_layout.addWidget(metrics_group)

        # 학습 loss curve
        loss_group = QGroupBox("학습 Loss Curve")
        loss_layout = QVBoxLayout(loss_group)
        from naviertwin.gui.widgets.loss_curve_widget import LossCurveWidget

        self._loss_curve = LossCurveWidget()
        loss_layout.addWidget(self._loss_curve)
        right_layout.addWidget(loss_group)

        # 로그
        log_group = QGroupBox("로그")
        log_layout = QVBoxLayout(log_group)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        log_layout.addWidget(self._log_text)
        right_layout.addWidget(log_group)

        layout.addWidget(right, stretch=1)

    # ──────────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────────

    def set_reducer(self, reducer: object) -> None:
        """차원 축소 결과를 설정한다."""
        self._reducer = reducer
        self._log(f"Reducer 설정: {type(reducer).__name__}")

    def set_dataset(self, dataset: CFDDataset) -> None:
        """학습에 사용할 원본 데이터셋을 설정한다."""
        self._dataset = dataset
        self._reduction_artifact = None
        self._log(
            f"Dataset 설정: {dataset.n_points} pts, {dataset.n_cells} cells, "
            f"{dataset.n_time_steps} steps"
        )

    def set_reduction_artifact(self, artifact: dict[str, object]) -> None:
        """Reduce 단계 산출물을 설정한다."""
        self._reduction_artifact = artifact
        method = str(artifact.get("method", "unknown"))
        field_name = str(artifact.get("field_name", ""))
        self._log(f"Reduction artifact 설정: method={method}, field={field_name}")

    def set_loaded_metadata(self, meta: dict[str, object]) -> None:
        """프로젝트 로드 시 저장된 메타데이터로 UI 상태를 복원한다."""
        self._loaded_metadata = dict(meta)

        engine_meta = meta.get("engine")
        surrogate_meta: dict[str, object] = {}
        if isinstance(engine_meta, dict):
            candidate = engine_meta.get("surrogate")
            if isinstance(candidate, dict):
                surrogate_meta = candidate

        reducer_meta: dict[str, object] = {}
        if isinstance(engine_meta, dict):
            candidate = engine_meta.get("reducer")
            if isinstance(candidate, dict):
                reducer_meta = candidate

        restored: list[str] = []

        n_modes = (
            self._as_positive_int(meta.get("n_modes"))
            or self._as_positive_int(engine_meta.get("n_modes") if isinstance(engine_meta, dict) else None)
            or self._as_positive_int(surrogate_meta.get("n_modes"))
            or self._as_positive_int(reducer_meta.get("n_modes"))
        )
        if n_modes is not None:
            self._n_outputs_spin.setValue(n_modes)
            restored.append(f"n_modes={n_modes}")

        n_params = (
            self._as_positive_int(meta.get("n_params"))
            or self._as_positive_int(surrogate_meta.get("n_params"))
        )
        if n_params is not None:
            self._n_params_spin.setValue(n_params)
            restored.append(f"n_params={n_params}")

        n_outputs = (
            self._as_positive_int(meta.get("n_outputs"))
            or self._as_positive_int(surrogate_meta.get("n_outputs"))
            or self._as_positive_int(surrogate_meta.get("n_modes"))
            or self._as_positive_int(engine_meta.get("n_modes") if isinstance(engine_meta, dict) else None)
        )
        if n_outputs is not None:
            self._n_outputs_spin.setValue(n_outputs)
            restored.append(f"n_outputs={n_outputs}")

        n_samples = (
            self._as_positive_int(meta.get("n_samples"))
            or self._as_positive_int(surrogate_meta.get("n_samples"))
        )
        if n_samples is not None:
            self._n_samples_spin.setValue(n_samples)
            restored.append(f"n_samples={n_samples}")

        if restored:
            self._log("Loaded metadata restored: " + ", ".join(restored))
        else:
            self._log("Loaded metadata restored: no numeric fields found")

    # ──────────────────────────────────────────────────────────────────
    # 슬롯
    # ──────────────────────────────────────────────────────────────────

    def _train_model(self) -> None:
        model_idx = self._model_combo.currentIndex()
        n_samples = self._n_samples_spin.value()
        n_params = self._n_params_spin.value()
        n_outputs = self._n_outputs_spin.value()
        ratio = self._train_ratio_spin.value()

        try:
            X: np.ndarray
            Y: np.ndarray
            used_real_data = False

            # 최우선: Reduce 단계 산출물(coeffs/params) 사용
            if self._reduction_artifact is not None:
                coeffs_raw = self._reduction_artifact.get("coeffs")
                params_raw = self._reduction_artifact.get("params")
                if coeffs_raw is not None and params_raw is not None:
                    coeffs = np.asarray(coeffs_raw, dtype=float)
                    params = np.asarray(params_raw, dtype=float)
                    if params.ndim == 1:
                        params = params.reshape(-1, 1)
                    if coeffs.ndim == 1:
                        coeffs = coeffs.reshape(-1, 1)
                    if coeffs.shape[0] == params.shape[0] and coeffs.shape[0] >= 2:
                        X = params
                        Y = coeffs
                        used_real_data = True
                        n_samples = X.shape[0]
                        n_params = X.shape[1]
                        n_outputs = Y.shape[1]
                        self._n_params_spin.setValue(n_params)
                        self._n_outputs_spin.setValue(n_outputs)
                        self._log(
                            f"artifact 기반 학습 사용: samples={n_samples}, "
                            f"params={n_params}, outputs={n_outputs}"
                        )

            # 우선: reducer + dataset이 있으면 실제 스냅샷 기반 학습 데이터 사용
            if (
                not used_real_data
                and self._reducer is not None
                and self._dataset is not None
                and self._dataset.field_names
            ):
                try:
                    field_name = self._dataset.field_names[0]
                    reducer_meta = getattr(self._reducer, "training_metadata", None)
                    if isinstance(reducer_meta, dict):
                        meta_field = reducer_meta.get("field_name")
                        if (
                            isinstance(meta_field, str)
                            and meta_field in self._dataset.field_names
                        ):
                            field_name = meta_field
                    snapshots = self._extract_snapshots_from_dataset(field_name)
                    if snapshots.shape[1] >= 2:
                        coeffs = self._reducer.encode(snapshots)
                        # 기본 파라미터는 time step 사용
                        ts = np.asarray(self._dataset.time_steps, dtype=float)
                        if ts.size == coeffs.shape[0]:
                            params_1d = ts
                        else:
                            params_1d = np.linspace(0.0, 1.0, coeffs.shape[0])
                        X = params_1d.reshape(-1, 1)
                        Y = coeffs
                        used_real_data = True
                        n_samples = X.shape[0]
                        n_params = X.shape[1]
                        n_outputs = Y.shape[1]
                        self._n_params_spin.setValue(n_params)
                        self._n_outputs_spin.setValue(n_outputs)
                        self._log(
                            f"실데이터 학습 사용: field={field_name}, samples={n_samples}, "
                            f"params={n_params}, outputs={n_outputs}"
                        )
                except Exception as exc:
                    self._log(f"[WARN] 실데이터 학습 데이터 구성 실패, 데모로 폴백: {exc}")

            if not used_real_data:
                if self._reducer is not None:
                    reducer_modes = int(getattr(self._reducer, "n_components", n_outputs))
                    if reducer_modes > 0:
                        n_outputs = reducer_modes
                        self._n_outputs_spin.setValue(n_outputs)
                        self._log(f"Reducer 모드 수에 맞춰 출력 차원을 {n_outputs}로 설정")

                # 데모: 랜덤 데이터 생성
                rng = np.random.default_rng(42)
                X = rng.random((n_samples, n_params))
                Y = np.sin(X.sum(axis=1, keepdims=True) * np.pi) * rng.random(
                    (n_samples, n_outputs)
                )
                self._log("데모 데이터로 학습 수행")

            n_train = max(2, int(n_samples * ratio))
            X_train, X_test = X[:n_train], X[n_train:]
            Y_train, Y_test = Y[:n_train], Y[n_train:]

            if model_idx <= 0 or model_idx == 2:
                from naviertwin.core.surrogate.kriging_surrogate import KrigingSurrogate
                surrogate = KrigingSurrogate()
            else:
                from naviertwin.core.surrogate.rbf_surrogate import RBFSurrogate
                surrogate = RBFSurrogate()

            surrogate.fit(X_train, Y_train)
            self._surrogate = surrogate
            model_name = type(surrogate).__name__
            surrogate.training_metadata = {
                "dataset_id": id(self._dataset) if self._dataset is not None else None,
                "field_name": (
                    self._dataset.field_names[0]
                    if self._dataset and self._dataset.field_names
                    else ""
                ),
                "n_modes": int(Y.shape[1]),
                "n_outputs": int(Y.shape[1]),
                "n_params": int(X.shape[1]),
                "n_samples": int(Y.shape[0]),
                "source": "real_data" if used_real_data else "demo",
            }

            # 검증
            metrics_text = ""
            validation_metrics: dict[str, float] = {}
            if len(X_test) > 0:
                from naviertwin.core.validation.metrics import compute_all_metrics
                Y_pred = surrogate.predict(X_test)
                metrics = compute_all_metrics(Y_test, Y_pred)
                validation_metrics = {
                    k: float(v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v))
                }
                self._metrics_list.clear()
                for k, v in metrics.items():
                    self._metrics_list.addItem(f"{k}: {v:.6f}")
                metrics_text = f"  RMSE={metrics.get('rmse', 0):.4g}, R²={metrics.get('r2', 0):.4g}"
            else:
                self._metrics_list.clear()
                self._metrics_list.addItem("테스트 셋 없음 (샘플 수 증가 필요)")
            surrogate.training_metadata["validation_metrics"] = validation_metrics

            self._log(f"{model_name} 학습 완료 (n_train={n_train}){metrics_text}")
            self.model_trained.emit(model_name.lower(), surrogate)

        except Exception as exc:
            self._log(f"[ERROR] {exc}")

    def _train_operator(self) -> None:
        """선택한 신경 연산자를 합성 데이터로 빠르게 학습/검증한다."""
        import numpy as np

        op_type = self._op_combo.currentText()
        epochs = int(self._op_epochs_spin.value())
        n = int(self._op_samples_spin.value())
        rng = np.random.default_rng(0)

        try:
            if op_type == "FNO1D":
                from naviertwin.core.operator_learning.fno.fno import FNO1D

                X = rng.standard_normal((n, 32, 1)).astype(np.float32)
                Y = np.sin(X).astype(np.float32)
                op = FNO1D(
                    in_channels=1, out_channels=1, modes=4, width=8,
                    n_layers=2, max_epochs=epochs,
                )
                op.fit({"inputs": X, "outputs": Y})
                y_hat = op.predict({"x": X[:2]})
                msg = f"FNO1D: 예측 shape={y_hat.shape}, 마지막 loss={op.train_losses_[-1]:.4g}"

            elif op_type == "FNO2D":
                from naviertwin.core.operator_learning.fno.fno import FNO2D

                X = rng.standard_normal((max(n // 2, 4), 16, 16, 1)).astype(np.float32)
                Y = X ** 2
                op = FNO2D(
                    in_channels=1, out_channels=1, modes1=4, modes2=4,
                    width=8, n_layers=2, max_epochs=epochs,
                )
                op.fit({"inputs": X, "outputs": Y})
                y_hat = op.predict({"x": X[:2]})
                msg = f"FNO2D: shape={y_hat.shape}, loss={op.train_losses_[-1]:.4g}"

            elif op_type == "TFNO2D":
                from naviertwin.core.operator_learning.fno.tfno import TFNO2D

                X = rng.standard_normal((max(n // 2, 4), 16, 16, 1)).astype(np.float32)
                Y = X ** 2
                op = TFNO2D(
                    in_channels=1, out_channels=1, modes1=4, modes2=4,
                    width=8, rank=4, n_layers=2, max_epochs=epochs,
                )
                op.fit({"inputs": X, "outputs": Y})
                y_hat = op.predict({"x": X[:2]})
                msg = (
                    f"TFNO2D: shape={y_hat.shape}, params={op.param_count()}, "
                    f"loss={op.train_losses_[-1]:.4g}"
                )

            elif op_type == "DeepONet":
                from naviertwin.core.operator_learning.deeponet.deeponet import DeepONet

                m, q = 16, 10
                Bx = rng.standard_normal((n, m)).astype(np.float32)
                Tx = rng.standard_normal((q, 2)).astype(np.float32)
                Y = np.tanh(Bx @ rng.standard_normal((m, q)).astype(np.float32))
                op = DeepONet(
                    branch_in=m, trunk_in=2, hidden=16, latent=8, max_epochs=epochs,
                )
                op.fit({"branch_inputs": Bx, "trunk_inputs": Tx, "outputs": Y})
                y_hat = op.predict({"branch_inputs": Bx[:2]})
                msg = f"DeepONet: shape={y_hat.shape}, loss={op.train_losses_[-1]:.4g}"

            elif op_type == "UNet2D":
                from naviertwin.core.operator_learning.unet.unet import UNet2D

                X = rng.standard_normal((max(n // 2, 4), 16, 16, 1)).astype(np.float32)
                Y = X ** 2
                op = UNet2D(
                    in_channels=1, out_channels=1, base_ch=8, max_epochs=epochs,
                )
                op.fit({"inputs": X, "outputs": Y})
                y_hat = op.predict({"x": X[:2]})
                msg = f"UNet2D: shape={y_hat.shape}, loss={op.train_losses_[-1]:.4g}"

            elif op_type == "WNO1D":
                try:
                    from naviertwin.core.operator_learning.fno.wno import WNO1D
                except Exception as exc:  # noqa: BLE001
                    self._log(f"[WARN] WNO1D import 실패: {exc}")
                    return

                X = rng.standard_normal((n, 64, 1)).astype(np.float32)
                Y = X ** 2
                op = WNO1D(
                    in_channels=1, out_channels=1, width=8,
                    wavelet="db2", level=2, n_layers=2, max_epochs=epochs,
                )
                try:
                    op.fit({"inputs": X, "outputs": Y})
                except RuntimeError as exc:
                    self._log(f"[WARN] WNO1D 학습 실패 (pywt 필요): {exc}")
                    return
                y_hat = op.predict({"x": X[:2]})
                msg = f"WNO1D: shape={y_hat.shape}, loss={op.train_losses_[-1]:.4g}"

            else:
                self._log(f"[WARN] 알 수 없는 연산자: {op_type}")
                return

            self._log(f"[{op_type}] 학습 완료 — {msg}")
            self._update_loss_curve(op_type, op)
            self.model_trained.emit(op_type.lower(), op)

        except Exception as exc:  # noqa: BLE001
            self._log(f"[ERROR] {op_type} 학습 실패: {exc}")

    def _extract_snapshots_from_dataset(self, field: str) -> np.ndarray:
        """CFDDataset에서 (n_features, n_samples) 스냅샷 행렬을 구성한다."""
        if self._dataset is None:
            raise RuntimeError("dataset이 설정되지 않았습니다.")
        return self._dataset.extract_field_snapshots(field)

    @staticmethod
    def _as_positive_int(value: object) -> int | None:
        """양의 정수로 해석 가능한 값을 반환한다."""
        try:
            number = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    def _update_loss_curve(self, label: str, model: object) -> None:
        """모델의 train_losses_를 LossCurveWidget에 반영한다."""
        raw_losses = getattr(model, "train_losses_", None)
        if raw_losses is None:
            return
        try:
            losses = [
                float(value)
                for value in raw_losses
                if np.isfinite(float(value))
            ]
        except (TypeError, ValueError):
            return
        if not losses:
            return
        self._loss_series[label] = losses
        self._loss_curve.set_losses(self._loss_series)
        self._log(f"Loss curve 업데이트: {label} ({len(losses)} epochs)")

    def _log(self, msg: str) -> None:
        self._log_text.append(msg)
        sb = self._log_text.verticalScrollBar()
        sb.setValue(sb.maximum())
