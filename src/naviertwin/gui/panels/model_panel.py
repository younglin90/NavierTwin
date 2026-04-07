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
    pass


class ModelPanel(QWidget):
    """Surrogate 모델 학습 패널.

    Signals:
        model_trained: 모델 학습 완료 시 surrogate 객체와 함께 발생.
    """

    model_trained = Signal(str, object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._reducer: Optional[object] = None
        self._surrogate: Optional[object] = None
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
            # 데모: 랜덤 데이터 생성
            rng = np.random.default_rng(42)
            X = rng.random((n_samples, n_params))
            Y = np.sin(X.sum(axis=1, keepdims=True) * np.pi) * rng.random((n_samples, n_outputs))

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

            # 검증
            metrics_text = ""
            if len(X_test) > 0:
                from naviertwin.core.validation.metrics import compute_all_metrics
                Y_pred = surrogate.predict(X_test)
                metrics = compute_all_metrics(Y_test, Y_pred)
                self._metrics_list.clear()
                for k, v in metrics.items():
                    self._metrics_list.addItem(f"{k}: {v:.6f}")
                metrics_text = f"  RMSE={metrics.get('rmse', 0):.4g}, R²={metrics.get('r2', 0):.4g}"
            else:
                self._metrics_list.clear()
                self._metrics_list.addItem("테스트 셋 없음 (샘플 수 증가 필요)")

            self._log(f"{model_name} 학습 완료 (n_train={n_train}){metrics_text}")
            self.model_trained.emit(model_name.lower(), surrogate)

        except Exception as exc:
            self._log(f"[ERROR] {exc}")

    def _log(self, msg: str) -> None:
        self._log_text.append(msg)
        sb = self._log_text.verticalScrollBar()
        sb.setValue(sb.maximum())
