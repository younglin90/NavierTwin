"""디지털 트윈 예측 패널 — TwinEngine 을 통한 파라미터 → 필드 예측.

Signals:
    prediction_done(object): 예측 완료 시 복원된 필드 배열 발생.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class TwinPanel(QWidget):
    """디지털 트윈 탭 패널.

    Signals:
        prediction_done: 예측 완료 시 결과 배열과 함께 발생.
    """

    prediction_done = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._engine: Optional[object] = None
        self._n_params: int = 2
        self._param_spins: list[QDoubleSpinBox] = []
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
        left.setFixedWidth(300)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        title = QLabel("Digital Twin Prediction")
        title.setObjectName("titleLabel")
        left_layout.addWidget(title)

        # 엔진 설정
        engine_group = QGroupBox("TwinEngine 설정")
        engine_layout = QFormLayout(engine_group)

        reducer_row = QHBoxLayout()
        from PySide6.QtWidgets import QComboBox
        self._reducer_combo = QComboBox()
        self._reducer_combo.addItems(["pod", "randomized_pod"])
        reducer_row.addWidget(self._reducer_combo)
        engine_layout.addRow("Reducer:", self._reducer_combo)

        self._surrogate_combo = QComboBox()
        self._surrogate_combo.addItems(["kriging", "rbf"])
        engine_layout.addRow("Surrogate:", self._surrogate_combo)

        self._n_modes_spin = QSpinBox()
        self._n_modes_spin.setRange(1, 200)
        self._n_modes_spin.setValue(10)
        engine_layout.addRow("POD 모드 수:", self._n_modes_spin)

        left_layout.addWidget(engine_group)

        # 파라미터 입력
        param_group = QGroupBox("파라미터 입력")
        param_outer_layout = QVBoxLayout(param_group)

        n_params_row = QHBoxLayout()
        n_params_row.addWidget(QLabel("파라미터 수:"))
        self._n_params_spin = QSpinBox()
        self._n_params_spin.setRange(1, 20)
        self._n_params_spin.setValue(2)
        self._n_params_spin.valueChanged.connect(self._rebuild_param_inputs)
        n_params_row.addWidget(self._n_params_spin)
        param_outer_layout.addLayout(n_params_row)

        # 스크롤 가능한 파라미터 영역
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        self._param_widget = QWidget()
        self._param_layout = QFormLayout(self._param_widget)
        scroll.setWidget(self._param_widget)
        param_outer_layout.addWidget(scroll)

        left_layout.addWidget(param_group)

        # 엔진 로드/저장
        io_group = QGroupBox("엔진 저장/로드")
        io_layout = QHBoxLayout(io_group)

        self._save_btn = QPushButton("저장")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_engine)
        io_layout.addWidget(self._save_btn)

        self._load_btn = QPushButton("로드")
        self._load_btn.clicked.connect(self._load_engine)
        io_layout.addWidget(self._load_btn)

        left_layout.addWidget(io_group)

        # 예측 버튼
        self._predict_btn = QPushButton("예측 실행")
        self._predict_btn.setObjectName("primaryButton")
        self._predict_btn.clicked.connect(self._run_predict)
        left_layout.addWidget(self._predict_btn)

        # 데모 학습 버튼
        self._demo_btn = QPushButton("데모 학습 & 예측")
        self._demo_btn.clicked.connect(self._run_demo)
        left_layout.addWidget(self._demo_btn)

        left_layout.addStretch()
        layout.addWidget(left)

        # 우측: 결과
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        result_group = QGroupBox("예측 결과")
        result_layout = QVBoxLayout(result_group)
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        result_layout.addWidget(self._result_text)
        right_layout.addWidget(result_group)

        self._status_label = QLabel("TwinEngine을 로드하거나 데모를 실행하세요.")
        self._status_label.setObjectName("subtitleLabel")
        right_layout.addWidget(self._status_label)

        layout.addWidget(right, stretch=1)

        # 초기 파라미터 입력 빌드
        self._rebuild_param_inputs(2)

    def _rebuild_param_inputs(self, n: int) -> None:
        """파라미터 수에 맞게 입력 스핀박스를 재생성한다."""
        # 기존 위젯 제거
        while self._param_layout.rowCount() > 0:
            self._param_layout.removeRow(0)
        self._param_spins.clear()

        self._n_params = n
        for i in range(n):
            spin = QDoubleSpinBox()
            spin.setRange(-1e6, 1e6)
            spin.setValue(0.5)
            spin.setDecimals(4)
            spin.setSingleStep(0.1)
            self._param_layout.addRow(f"param_{i}:", spin)
            self._param_spins.append(spin)

    # ──────────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────────

    def set_engine(self, engine: object) -> None:
        """외부에서 TwinEngine을 설정한다."""
        self._engine = engine
        self._save_btn.setEnabled(True)
        self._log(f"TwinEngine 설정: {type(engine).__name__}")

    # ──────────────────────────────────────────────────────────────────
    # 슬롯
    # ──────────────────────────────────────────────────────────────────

    def _run_predict(self) -> None:
        if self._engine is None:
            self._log("[WARN] TwinEngine이 없습니다. 먼저 로드하거나 데모를 실행하세요.")
            return
        try:
            params = np.array([[spin.value() for spin in self._param_spins]], dtype=float)
            result = self._engine.predict(params)  # type: ignore[union-attr]
            self._log(f"예측 완료: shape={result.shape}, min={result.min():.4g}, max={result.max():.4g}")
            self._status_label.setText("예측 완료.")
            self.prediction_done.emit(result)
        except Exception as exc:
            self._log(f"[ERROR] {exc}")

    def _run_demo(self) -> None:
        """데모: 랜덤 데이터로 TwinEngine을 학습하고 예측한다."""
        try:
            from naviertwin.core.digital_twin.twin_engine import TwinEngine

            n_params = self._n_params_spin.value()
            n_modes = self._n_modes_spin.value()
            reducer = self._reducer_combo.currentText()
            surrogate = self._surrogate_combo.currentText()

            # 데모 데이터
            rng = np.random.default_rng(0)
            n_features = 200
            n_samples = 30
            snapshots = rng.random((n_features, n_samples))
            params = rng.random((n_samples, n_params))

            engine = TwinEngine(
                reducer_type=reducer,
                surrogate_type=surrogate,
                n_modes=min(n_modes, n_samples - 1),
            )
            engine.fit(snapshots, params)
            self._engine = engine
            self._save_btn.setEnabled(True)

            # 예측
            test_params = np.array([[spin.value() for spin in self._param_spins]], dtype=float)
            if test_params.shape[1] != n_params:
                test_params = rng.random((1, n_params))

            result = engine.predict(test_params)
            self._log(
                f"[Demo] TwinEngine 학습+예측 완료\n"
                f"  reducer={reducer}, surrogate={surrogate}, n_modes={n_modes}\n"
                f"  output shape={result.shape}, range=[{result.min():.4g}, {result.max():.4g}]\n"
            )
            self._status_label.setText("데모 완료.")
            self.prediction_done.emit(result)

        except Exception as exc:
            self._log(f"[ERROR] {exc}")

    def _save_engine(self) -> None:
        if self._engine is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "TwinEngine 저장", "twin_engine.pkl", "Pickle (*.pkl)"
        )
        if path:
            try:
                self._engine.save(Path(path))  # type: ignore[union-attr]
                self._log(f"엔진 저장: {path}")
            except Exception as exc:
                self._log(f"[ERROR] 저장 실패: {exc}")

    def _load_engine(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "TwinEngine 로드", "", "Pickle (*.pkl)"
        )
        if path:
            try:
                from naviertwin.core.digital_twin.twin_engine import TwinEngine
                engine = TwinEngine.load(Path(path))
                self.set_engine(engine)
                self._log(f"엔진 로드: {path}")
            except Exception as exc:
                self._log(f"[ERROR] 로드 실패: {exc}")

    def _log(self, msg: str) -> None:
        self._result_text.append(msg)
        sb = self._result_text.verticalScrollBar()
        sb.setValue(sb.maximum())
