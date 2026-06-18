"""Explainability panel used with trained surrogate and attention models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ExplainabilityPanel(QWidget):
    """SHAP and attention explainability tab used with GUI-trained models."""

    explanation_done = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._dataset: object | None = None
        self._model: object | None = None
        self._background: np.ndarray | None = None
        self._feature_names: list[str] = []
        self._attention_source: object | None = None
        self._attention_source_name = ""
        self._attention_probe: np.ndarray | None = None
        self._attention_token_names: list[str] = []
        self._setup_ui()
        self._refresh_enabled()

    def _setup_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer_layout.addWidget(scroll)

        content = QWidget(scroll)
        scroll.setWidget(content)
        self._scroll_area = scroll

        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Explainability")
        title.setObjectName("titleLabel")
        layout.addWidget(title)

        subtitle = QLabel(
            "학습된 surrogate의 SHAP/심볼릭 식과 attention 모델의 token 가중치를 설명합니다."
        )
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

        symbolic_group = QGroupBox("Symbolic Regression 옵션")
        symbolic_form = QFormLayout(symbolic_group)
        self._symbolic_degree_spin = QSpinBox()
        self._symbolic_degree_spin.setRange(1, 5)
        self._symbolic_degree_spin.setValue(3)
        symbolic_form.addRow("Max degree:", self._symbolic_degree_spin)

        self._symbolic_threshold_spin = QDoubleSpinBox()
        self._symbolic_threshold_spin.setRange(0.0, 1.0)
        self._symbolic_threshold_spin.setDecimals(6)
        self._symbolic_threshold_spin.setSingleStep(0.001)
        self._symbolic_threshold_spin.setValue(0.001)
        symbolic_form.addRow("Coefficient threshold:", self._symbolic_threshold_spin)
        layout.addWidget(symbolic_group)

        symbolic_btn_row = QHBoxLayout()
        symbolic_btn_row.addStretch()
        self._symbolic_btn = QPushButton("Symbolic 식 추정")
        self._symbolic_btn.clicked.connect(self._run_symbolic)
        symbolic_btn_row.addWidget(self._symbolic_btn)
        layout.addLayout(symbolic_btn_row)

        attention_group = QGroupBox("Attention 옵션")
        attention_form = QFormLayout(attention_group)
        self._attention_source_label = QLabel("감지 안됨")
        attention_form.addRow("Attention module:", self._attention_source_label)

        self._attention_batch_spin = QSpinBox()
        self._attention_batch_spin.setRange(1, 128)
        self._attention_batch_spin.setValue(1)
        attention_form.addRow("Batch:", self._attention_batch_spin)

        self._attention_tokens_spin = QSpinBox()
        self._attention_tokens_spin.setRange(1, 2048)
        self._attention_tokens_spin.setValue(8)
        attention_form.addRow("Tokens:", self._attention_tokens_spin)

        self._attention_dim_spin = QSpinBox()
        self._attention_dim_spin.setRange(1, 65536)
        self._attention_dim_spin.setValue(64)
        attention_form.addRow("Embedding dim:", self._attention_dim_spin)

        self._attention_topk_spin = QSpinBox()
        self._attention_topk_spin.setRange(1, 32)
        self._attention_topk_spin.setValue(3)
        attention_form.addRow("Top-k keys:", self._attention_topk_spin)
        layout.addWidget(attention_group)

        attention_btn_row = QHBoxLayout()
        attention_btn_row.addStretch()
        self._attention_btn = QPushButton("Attention 시각화 실행")
        self._attention_btn.clicked.connect(self._run_attention)
        attention_btn_row.addWidget(self._attention_btn)
        layout.addLayout(attention_btn_row)

        table_group = QGroupBox("Feature 기여도")
        table_layout = QVBoxLayout(table_group)
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Feature", "Mean SHAP", "Mean |SHAP|"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table_layout.addWidget(self._table)
        layout.addWidget(table_group, stretch=1)

        symbolic_result_group = QGroupBox("Symbolic expression")
        symbolic_result_layout = QVBoxLayout(symbolic_result_group)
        self._symbolic_text = QTextEdit()
        self._symbolic_text.setReadOnly(True)
        symbolic_result_layout.addWidget(self._symbolic_text)
        layout.addWidget(symbolic_result_group)

        attention_result_group = QGroupBox("Attention weights")
        attention_result_layout = QVBoxLayout(attention_result_group)
        self._attention_matrix_table = QTableWidget(0, 0)
        self._attention_matrix_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        attention_result_layout.addWidget(self._attention_matrix_table)
        self._attention_top_table = QTableWidget(0, 3)
        self._attention_top_table.setHorizontalHeaderLabels(
            ["Query token", "Top key tokens", "Max weight"]
        )
        self._attention_top_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        attention_result_layout.addWidget(self._attention_top_table)
        layout.addWidget(attention_result_group, stretch=1)

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
        self._attention_source = None
        self._attention_source_name = ""
        self._attention_probe = None
        self._attention_token_names = []
        metadata = getattr(model, "training_metadata", None)
        if not isinstance(metadata, Mapping):
            self._log("[WARN] 모델에 explainability metadata가 없습니다.")
            self._load_attention_metadata(model, None)
            return
        explain_meta = metadata.get("explainability")
        if not isinstance(explain_meta, Mapping):
            self._log("[WARN] 모델에 explainability background가 없습니다.")
        else:
            background = np.asarray(explain_meta.get("background"), dtype=float)
            if background.ndim != 2 or background.shape[0] == 0 or background.shape[1] == 0:
                self._log("[WARN] explainability background shape이 유효하지 않습니다.")
            else:
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

        self._load_attention_metadata(model, metadata.get("attention"))

    def _load_attention_metadata(
        self, model: object, attention_meta: object | None,
    ) -> None:
        source = None
        source_name = ""
        if isinstance(attention_meta, Mapping):
            path = attention_meta.get("module_path") or attention_meta.get("path")
            if path is not None:
                source = self._resolve_attr_path(model, str(path))
                source_name = str(path)
            if source is None:
                keys = ("module", "attention_module", "mha", "source")
                key_index = 0
                while key_index < len(keys):
                    key = keys[key_index]
                    value = attention_meta.get(key)
                    if value is not None:
                        source = value
                        source_name = key
                        break
                    key_index += 1
            probe_value = attention_meta.get("probe")
            if probe_value is None:
                probe_value = attention_meta.get("input")
            self._attention_probe = self._coerce_attention_probe(probe_value)
            if self._attention_probe is not None:
                self._apply_probe_shape_to_attention_options(self._attention_probe)
            token_count = self._attention_tokens_spin.value()
            self._attention_token_names = self._normalize_feature_names(
                attention_meta.get("token_names"), token_count,
            )

        if source is None:
            source, source_name = self._find_first_multihead_attention(model)

        if source is None:
            self._attention_source_label.setText("감지 안됨")
            return

        self._attention_source = source
        self._attention_source_name = source_name or type(source).__name__
        self._attention_source_label.setText(self._attention_source_name)
        embed_dim = self._safe_int(
            getattr(source, "embed_dim", None),
            default=self._attention_dim_spin.value(),
        )
        self._attention_dim_spin.setValue(max(1, embed_dim))
        if not self._attention_token_names:
            self._attention_token_names = self._normalize_feature_names(
                None, self._attention_tokens_spin.value(),
            )
        self._log(f"Attention module 설정: {self._attention_source_name}")

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

    def _run_symbolic(self) -> None:
        if self._model is None or self._background is None:
            self._log("[WARN] Symbolic regression을 실행할 모델/background가 없습니다.")
            return

        try:
            from naviertwin.core.explainability.symbolic_regression import (
                SymbolicRegressor,
            )

            X = self._background
            y = self._predict_scalar(X)
            regressor = SymbolicRegressor(
                max_degree=int(self._symbolic_degree_spin.value()),
                threshold=float(self._symbolic_threshold_spin.value()),
            )
            regressor.fit(X, y)
            expression = regressor.expression_
            self._symbolic_text.setPlainText(expression)
            result = {
                "symbolic_expression": expression,
                "max_degree": int(self._symbolic_degree_spin.value()),
                "threshold": float(self._symbolic_threshold_spin.value()),
                "n_samples": int(X.shape[0]),
                "output_index": int(self._output_index_spin.value()),
            }
            self._log(f"Symbolic 완료: samples={X.shape[0]}, expr={expression}")
            self.explanation_done.emit(result)
        except Exception as exc:
            self._log(f"[ERROR] Symbolic regression 실패: {exc}")

    def _run_attention(self) -> None:
        if self._attention_source is None:
            self._log("[WARN] Attention module이 감지되지 않았습니다.")
            return

        try:
            from naviertwin.core.explainability.attention_viz import (
                extract_attention,
                topk_attention_tokens,
            )

            source = self._attention_source
            training = bool(getattr(source, "training", False))
            eval_fn = getattr(source, "eval", None)
            if callable(eval_fn):
                eval_fn()
            x = self._make_attention_input(source)
            try:
                import torch

                with torch.no_grad():
                    _, weights = extract_attention(source, x)
            finally:
                train_fn = getattr(source, "train", None)
                if training and callable(train_fn):
                    train_fn()

            weights = np.asarray(weights, dtype=float)
            top_tokens = topk_attention_tokens(
                weights,
                k=min(self._attention_topk_spin.value(), weights.shape[-1]),
            )
            self._render_attention(weights, top_tokens)
            result = {
                "attention_weights": weights,
                "top_tokens": top_tokens,
                "token_names": list(self._attention_token_names),
                "source": self._attention_source_name,
            }
            self._log(
                f"Attention 완료: source={self._attention_source_name}, "
                f"shape={weights.shape}"
            )
            self.explanation_done.emit(result)
        except Exception as exc:
            self._log(f"[ERROR] Attention 실패: {exc}")

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
        row = 0
        while row < phi.shape[1]:
            feature = self._feature_names[row] if row < len(self._feature_names) else f"param_{row}"
            self._table.setItem(row, 0, QTableWidgetItem(feature))
            self._table.setItem(row, 1, QTableWidgetItem(f"{mean_phi[row]:.6g}"))
            self._table.setItem(row, 2, QTableWidgetItem(f"{mean_abs[row]:.6g}"))
            row += 1

    def _render_attention(self, weights: np.ndarray, top_tokens: np.ndarray) -> None:
        if weights.ndim != 3:
            raise ValueError(f"Attention weights must be 3D, got shape={weights.shape}")
        mat = weights[0]
        max_rows = min(mat.shape[0], 32)
        max_cols = min(mat.shape[1], 32)
        row_names = self._attention_labels(max_rows)
        col_names = self._attention_labels(max_cols)

        self._attention_matrix_table.setRowCount(max_rows)
        self._attention_matrix_table.setColumnCount(max_cols)
        self._attention_matrix_table.setVerticalHeaderLabels(row_names)
        self._attention_matrix_table.setHorizontalHeaderLabels(col_names)
        row = 0
        while row < max_rows:
            col = 0
            while col < max_cols:
                self._attention_matrix_table.setItem(
                    row, col, QTableWidgetItem(f"{mat[row, col]:.4f}")
                )
                col += 1
            row += 1

        self._attention_top_table.setRowCount(max_rows)
        row = 0
        while row < max_rows:
            keys = []
            token_limit = min(top_tokens.shape[-1], max_cols)
            token_index = 0
            while token_index < token_limit:
                keys.append(int(top_tokens[0, row, token_index]))
                token_index += 1
            key_labels = []
            key_index = 0
            while key_index < len(keys):
                idx = keys[key_index]
                if idx < mat.shape[1]:
                    key_labels.append(
                        f"{self._attention_label_for(idx)} ({mat[row, idx]:.4f})"
                    )
                key_index += 1
            self._attention_top_table.setItem(
                row, 0, QTableWidgetItem(self._attention_label_for(row))
            )
            self._attention_top_table.setItem(
                row, 1, QTableWidgetItem(", ".join(key_labels))
            )
            self._attention_top_table.setItem(
                row, 2, QTableWidgetItem(f"{float(np.max(mat[row])):.4f}")
            )
            row += 1

    def _refresh_enabled(self) -> None:
        self._explain_btn.setEnabled(self._model is not None and self._background is not None)
        self._symbolic_btn.setEnabled(self._model is not None and self._background is not None)
        self._attention_btn.setEnabled(self._attention_source is not None)

    def _log(self, msg: str) -> None:
        self._log_text.append(msg)
        sb = self._log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    @staticmethod
    def _normalize_feature_names(value: object, n_features: int) -> list[str]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            names = list(map(str, value[:n_features]))
            if len(names) == n_features:
                return names
        names = []
        index = 0
        while index < n_features:
            names.append(f"param_{index}")
            index += 1
        return names

    @staticmethod
    def _safe_int(value: object, default: int) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _resolve_attr_path(root: object, path: str) -> object | None:
        current: object = root
        parts = path.split(".")
        part_index = 0
        while part_index < len(parts):
            part = parts[part_index]
            if not part:
                part_index += 1
                continue
            if part.isdigit() and hasattr(current, "__getitem__"):
                try:
                    current = current[int(part)]  # type: ignore[index]
                    part_index += 1
                    continue
                except (IndexError, KeyError, TypeError):
                    return None
            if isinstance(current, Mapping):
                current = current.get(part)
                if current is None:
                    return None
                part_index += 1
                continue
            if not hasattr(current, part):
                return None
            current = getattr(current, part)
            part_index += 1
        return current

    @staticmethod
    def _coerce_attention_probe(value: object) -> np.ndarray | None:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()  # type: ignore[union-attr]
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 3 or 0 in arr.shape:
            return None
        return arr

    def _apply_probe_shape_to_attention_options(self, probe: np.ndarray) -> None:
        self._attention_batch_spin.setValue(max(1, int(probe.shape[0])))
        self._attention_tokens_spin.setValue(max(1, int(probe.shape[1])))
        self._attention_dim_spin.setValue(max(1, int(probe.shape[2])))

    @staticmethod
    def _find_first_multihead_attention(model: object) -> tuple[object | None, str]:
        try:
            import torch.nn as nn
        except ImportError:
            return None, ""

        candidates = [
            ("model", model),
            ("model._model", getattr(model, "_model", None)),
            ("model.model", getattr(model, "model", None)),
            ("model.module", getattr(model, "module", None)),
            ("model.net", getattr(model, "net", None)),
            ("model.network", getattr(model, "network", None)),
        ]
        seen: set[int] = set()
        candidate_index = 0
        while candidate_index < len(candidates):
            prefix, candidate = candidates[candidate_index]
            if candidate is None or id(candidate) in seen:
                candidate_index += 1
                continue
            seen.add(id(candidate))
            if isinstance(candidate, nn.MultiheadAttention):
                return candidate, prefix
            modules = getattr(candidate, "modules", None)
            if callable(modules):
                module_list = tuple(modules())
                module_index = 0
                while module_index < len(module_list):
                    module = module_list[module_index]
                    if isinstance(module, nn.MultiheadAttention):
                        return module, f"{prefix}.{type(module).__name__}"
                    module_index += 1
            attrs = ("attn", "mha", "attention")
            attr_index = 0
            while attr_index < len(attrs):
                attr = attrs[attr_index]
                value = getattr(candidate, attr, None)
                if isinstance(value, nn.MultiheadAttention):
                    return value, f"{prefix}.{attr}"
                attr_index += 1
            candidate_index += 1
        return None, ""

    def _make_attention_input(self, source: object) -> object:
        import torch

        try:
            param = next(source.parameters())  # type: ignore[attr-defined]
            device = param.device
            dtype = param.dtype
        except (AttributeError, StopIteration):
            device = torch.device("cpu")
            dtype = torch.float32

        if self._attention_probe is not None:
            return torch.as_tensor(self._attention_probe, dtype=dtype, device=device)

        batch = self._attention_batch_spin.value()
        tokens = self._attention_tokens_spin.value()
        dim = self._attention_dim_spin.value()
        shape = (batch, tokens, dim) if bool(getattr(source, "batch_first", True)) else (
            tokens,
            batch,
            dim,
        )
        return torch.zeros(shape, dtype=dtype, device=device)

    def _attention_labels(self, count: int) -> list[str]:
        labels: list[str] = []
        index = 0
        while index < count:
            labels.append(self._attention_label_for(index))
            index += 1
        return labels

    def _attention_label_for(self, index: int) -> str:
        if index < len(self._attention_token_names):
            return self._attention_token_names[index]
        return f"token_{index}"


__all__ = ["ExplainabilityPanel"]
