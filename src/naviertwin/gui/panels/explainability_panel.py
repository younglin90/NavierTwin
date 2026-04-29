"""Explainability panel for trained surrogate models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ExplainabilityPanel(QWidget):
    """SHAP explainability tab for GUI-trained surrogate models."""

    explanation_done = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._dataset: object | None = None
        self._model: object | None = None
        self._background: np.ndarray | None = None
        self._feature_names: list[str] = []
        self._setup_ui()
        self._refresh_enabled()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Explainability")
        title.setObjectName("titleLabel")
        layout.addWidget(title)

        subtitle = QLabel("학습된 surrogate 모델의 파라미터 기여도를 Kernel SHAP으로 설명합니다.")
        subtitle.setObjectName("subtitleLabel")
        layout.addWidget(subtitle)

        source_group = QGroupBox("설명 대상")
        source_form = QFormLayout(source_group)
        self._model_label = QLabel("모델 없음")
        self._dataset_label = QLabel("데이터셋 없음")
        source_form.addRow("Model:", self._model_label)
        source_form.addRow("Dataset:", self._dataset_label)
        layout.addWidget(source_group)

        options_group = QGroupBox("SHAP 옵션")
        options_form = QFormLayout(options_group)
        self._output_index_spin = QSpinBox()
        self._output_index_spin.setRange(0, 999)
        self._output_index_spin.setValue(0)
        options_form.addRow("Output index:", self._output_index_spin)

        self._background_size_spin = QSpinBox()
        self._background_size_spin.setRange(1, 1024)
        self._background_size_spin.setValue(16)
        options_form.addRow("Background rows:", self._background_size_spin)

        self._sample_count_spin = QSpinBox()
        self._sample_count_spin.setRange(8, 5000)
        self._sample_count_spin.setValue(80)
        options_form.addRow("SHAP samples:", self._sample_count_spin)
        layout.addWidget(options_group)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._explain_btn = QPushButton("SHAP 설명 실행")
        self._explain_btn.setObjectName("primaryButton")
        self._explain_btn.clicked.connect(self._run_shap)
        btn_row.addWidget(self._explain_btn)
        layout.addLayout(btn_row)

        table_group = QGroupBox("Feature 기여도")
        table_layout = QVBoxLayout(table_group)
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Feature", "Mean SHAP", "Mean |SHAP|"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table_layout.addWidget(self._table)
        layout.addWidget(table_group, stretch=1)

        log_group = QGroupBox("로그")
        log_layout = QVBoxLayout(log_group)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        log_layout.addWidget(self._log_text)
        layout.addWidget(log_group, stretch=1)

    def set_dataset(self, dataset: object) -> None:
        """설명 결과 라벨링에 사용할 dataset context를 설정한다."""
        self._dataset = dataset
        n_points = getattr(dataset, "n_points", "?")
        n_cells = getattr(dataset, "n_cells", "?")
        self._dataset_label.setText(f"{n_points} pts, {n_cells} cells")

    def set_model(self, model: object) -> None:
        """학습된 surrogate/model과 explainability metadata를 설정한다."""
        self._model = model
        self._model_label.setText(type(model).__name__)
        self._load_explainability_metadata(model)
        self._refresh_enabled()

    def _load_explainability_metadata(self, model: object) -> None:
        self._background = None
        self._feature_names = []
        metadata = getattr(model, "training_metadata", None)
        if not isinstance(metadata, Mapping):
            self._log("[WARN] 모델에 explainability metadata가 없습니다.")
            return
        explain_meta = metadata.get("explainability")
        if not isinstance(explain_meta, Mapping):
            self._log("[WARN] 모델에 explainability background가 없습니다.")
            return

        background = np.asarray(explain_meta.get("background"), dtype=float)
        if background.ndim != 2 or background.shape[0] == 0 or background.shape[1] == 0:
            self._log("[WARN] explainability background shape이 유효하지 않습니다.")
            return

        self._background = background
        self._background_size_spin.setMaximum(max(1, background.shape[0]))
        self._background_size_spin.setValue(min(background.shape[0], 16))
        feature_names = self._normalize_feature_names(
            explain_meta.get("feature_names"),
            background.shape[1],
        )
        self._feature_names = feature_names
        output_index = self._safe_int(explain_meta.get("output_index"), default=0)
        self._output_index_spin.setValue(max(0, output_index))
        self._log(
            f"Explainability background 설정: rows={background.shape[0]}, "
            f"features={background.shape[1]}"
        )

    def _run_shap(self) -> None:
        if self._model is None or self._background is None:
            self._log("[WARN] SHAP을 실행할 모델/background가 없습니다.")
            return

        background_rows = min(self._background_size_spin.value(), self._background.shape[0])
        background = self._background[:background_rows]
        explain_x = background[: min(3, background.shape[0])]

        try:
            from naviertwin.core.explainability.shap_explainer import KernelSHAP

            explainer = KernelSHAP(
                self._predict_scalar,
                background=background,
                n_samples=int(self._sample_count_spin.value()),
                seed=0,
            )
            phi = np.asarray(explainer.explain(explain_x), dtype=float)
            self._render_phi(phi)
            result = {
                "phi": phi,
                "feature_names": list(self._feature_names),
                "output_index": int(self._output_index_spin.value()),
                "n_background": int(background.shape[0]),
            }
            self._log(
                f"SHAP 완료: samples={phi.shape[0]}, features={phi.shape[1]}, "
                f"background={background.shape[0]}"
            )
            self.explanation_done.emit(result)
        except Exception as exc:
            self._log(f"[ERROR] SHAP 실패: {exc}")

    def _predict_scalar(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None
        raw = self._model.predict(np.asarray(X, dtype=float))  # type: ignore[attr-defined]
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 0:
            return np.full((X.shape[0],), float(arr))
        if arr.ndim == 1:
            return arr
        flat = arr.reshape(arr.shape[0], -1)
        idx = min(int(self._output_index_spin.value()), flat.shape[1] - 1)
        return flat[:, idx]

    def _render_phi(self, phi: np.ndarray) -> None:
        if phi.ndim != 2:
            raise ValueError(f"SHAP result must be 2D, got shape={phi.shape}")
        mean_phi = np.mean(phi, axis=0)
        mean_abs = np.mean(np.abs(phi), axis=0)
        self._table.setRowCount(phi.shape[1])
        for row in range(phi.shape[1]):
            feature = self._feature_names[row] if row < len(self._feature_names) else f"param_{row}"
            self._table.setItem(row, 0, QTableWidgetItem(feature))
            self._table.setItem(row, 1, QTableWidgetItem(f"{mean_phi[row]:.6g}"))
            self._table.setItem(row, 2, QTableWidgetItem(f"{mean_abs[row]:.6g}"))

    def _refresh_enabled(self) -> None:
        self._explain_btn.setEnabled(self._model is not None and self._background is not None)

    def _log(self, msg: str) -> None:
        self._log_text.append(msg)
        sb = self._log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    @staticmethod
    def _normalize_feature_names(value: object, n_features: int) -> list[str]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            names = [str(item) for item in value[:n_features]]
            if len(names) == n_features:
                return names
        return [f"param_{i}" for i in range(n_features)]

    @staticmethod
    def _safe_int(value: object, default: int) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default


__all__ = ["ExplainabilityPanel"]
