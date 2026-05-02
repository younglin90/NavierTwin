"""Post-Processor Tools 패널 — Facade를 GUI에 노출.

Facade의 list_operations / describe / run을 호출해 사용자가 어떤 op든
선택해 실행 가능. 결과는 텍스트로 표시, 큰 배열은 요약 (mean/std/shape).

Signals:
    operation_done(str, object): op 실행 완료 시 (op_name, result_dict).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
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
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from naviertwin.core.post_process_facade import PostProcessFacade


class PostProcessPanel(QWidget):
    """후처리 도구 통합 패널 (R591-647 신규 모듈)."""

    operation_done = Signal(str, object)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        translator: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self._facade = PostProcessFacade()
        self._dataset = None
        # i18n translator (없으면 키 그대로 반환하는 fallback)
        if translator is None:
            try:
                from naviertwin.utils.i18n import Translator

                self._t = Translator(lang="ko")
            except Exception:  # noqa: BLE001
                self._t = lambda key: key  # type: ignore[assignment]
        else:
            self._t = translator
        self._setup_ui()

    def _tr(self, key: str) -> str:
        """Safe i18n lookup — fallback to key if translator missing."""
        try:
            return self._t(key) if callable(self._t) else self._t.t(key)
        except Exception:  # noqa: BLE001
            return key

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 좌측: 카테고리 + op 리스트
        left = QWidget()
        left.setFixedWidth(280)
        left_layout = QVBoxLayout(left)

        title = QLabel(self._tr("posttools.title"))
        title.setObjectName("titleLabel")
        left_layout.addWidget(title)

        self._data_label = QLabel("데이터: 합성 데모")
        self._data_label.setWordWrap(True)
        left_layout.addWidget(self._data_label)

        # 카테고리 필터
        cat_group = QGroupBox(self._tr("posttools.category"))
        cat_layout = QVBoxLayout(cat_group)
        self._category_combo = QComboBox()
        self._category_combo.addItem(self._tr("posttools.category.all"))
        cats = sorted({
            self._facade.describe(op)["category"]
            for op in self._facade.list_operations()
        })
        for c in cats:
            self._category_combo.addItem(c)
        self._category_combo.currentIndexChanged.connect(
            self._refresh_op_list,
        )
        cat_layout.addWidget(self._category_combo)
        left_layout.addWidget(cat_group)

        # op 리스트
        op_group = QGroupBox(self._tr("posttools.operations"))
        op_layout = QVBoxLayout(op_group)
        self._op_list = QListWidget()
        self._op_list.currentTextChanged.connect(self._on_op_selected)
        op_layout.addWidget(self._op_list)
        left_layout.addWidget(op_group)

        # 동적 파라미터 폼
        param_group = QGroupBox(self._tr("posttools.params"))
        param_group_layout = QVBoxLayout(param_group)
        # 프리셋 선택 콤보
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel(self._tr("posttools.preset")))
        self._preset_combo = QComboBox()
        self._preset_combo.currentTextChanged.connect(self._on_preset_selected)
        preset_row.addWidget(self._preset_combo, stretch=1)
        param_group_layout.addLayout(preset_row)
        # 동적 폼
        param_form_widget = QWidget()
        self._param_form_layout = QFormLayout(param_form_widget)
        self._param_widgets: dict[str, QWidget] = {}
        param_group_layout.addWidget(param_form_widget)
        left_layout.addWidget(param_group)

        # 프리셋 store
        from naviertwin.core.post_process_presets import PresetStore

        self._preset_store = PresetStore()

        # 실행 버튼
        self._run_btn = QPushButton(self._tr("posttools.run.demo"))
        self._run_btn.setObjectName("primaryButton")
        self._run_btn.clicked.connect(self._on_run_clicked)
        left_layout.addWidget(self._run_btn)

        # Export 버튼들
        export_row = QHBoxLayout()
        self._export_csv_btn = QPushButton("CSV")
        self._export_csv_btn.clicked.connect(self._on_export_csv)
        self._export_csv_btn.setEnabled(False)
        self._export_json_btn = QPushButton("JSON")
        self._export_json_btn.clicked.connect(self._on_export_json)
        self._export_json_btn.setEnabled(False)
        self._export_npz_btn = QPushButton("NPZ")
        self._export_npz_btn.clicked.connect(self._on_export_npz)
        self._export_npz_btn.setEnabled(False)
        export_row.addWidget(self._export_csv_btn)
        export_row.addWidget(self._export_json_btn)
        export_row.addWidget(self._export_npz_btn)
        left_layout.addLayout(export_row)

        # 차트 PNG 저장
        self._save_chart_btn = QPushButton(self._tr("posttools.save.chart"))
        self._save_chart_btn.clicked.connect(self._on_save_chart)
        self._save_chart_btn.setEnabled(False)
        left_layout.addWidget(self._save_chart_btn)

        # 카테고리 일괄 실행
        self._run_category_btn = QPushButton(self._tr("posttools.run.category"))
        self._run_category_btn.clicked.connect(self._on_run_category)
        left_layout.addWidget(self._run_category_btn)

        # 프리셋 저장 + 이력 viewer 버튼
        misc_row = QHBoxLayout()
        self._save_preset_btn = QPushButton(self._tr("posttools.save.preset"))
        self._save_preset_btn.clicked.connect(self._on_save_preset)
        self._show_history_btn = QPushButton(self._tr("posttools.show.history"))
        self._show_history_btn.clicked.connect(self._on_show_history)
        misc_row.addWidget(self._save_preset_btn)
        misc_row.addWidget(self._show_history_btn)
        left_layout.addLayout(misc_row)

        # 마지막 결과 저장 (export용)
        self._last_result: dict[str, Any] | None = None
        self._last_op_name: str | None = None

        # 실행 이력
        from naviertwin.core.post_process_history import RunHistory

        self._history = RunHistory(max_entries=50)

        left_layout.addStretch()
        layout.addWidget(left)

        # 우측: 설명 + 결과
        right_split = QSplitter()
        right_split.setOrientation(Qt.Orientation.Vertical)

        desc_group = QGroupBox(self._tr("posttools.description"))
        desc_layout = QFormLayout(desc_group)
        self._desc_label = QLabel("연산을 선택하세요.")
        self._desc_label.setWordWrap(True)
        self._params_label = QLabel("-")
        self._params_label.setWordWrap(True)
        self._returns_label = QLabel("-")
        self._returns_label.setWordWrap(True)
        desc_layout.addRow("Description:", self._desc_label)
        desc_layout.addRow("Parameters:", self._params_label)
        desc_layout.addRow("Returns:", self._returns_label)
        right_split.addWidget(desc_group)

        result_group = QGroupBox(self._tr("posttools.result"))
        result_layout = QVBoxLayout(result_group)
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setMaximumHeight(150)
        result_layout.addWidget(self._result_text)
        # 차트 위젯 (matplotlib 있으면)
        self._chart: QWidget | None = None
        try:
            from naviertwin.gui.widgets.postproc_chart import PostProcessChart

            self._chart = PostProcessChart()
            result_layout.addWidget(self._chart, stretch=1)
        except Exception:  # noqa: BLE001
            self._chart = None
        right_split.addWidget(result_group)

        right_split.setSizes([200, 600])
        layout.addWidget(right_split, stretch=1)

        # 초기화
        self._refresh_op_list()

    def set_dataset(self, dataset: object) -> None:
        """로드된 CFD dataset을 후처리 입력으로 연결한다."""
        self._dataset = dataset
        field_names = list(getattr(dataset, "field_names", []) or [])
        preview = ", ".join(field_names[:3]) if field_names else "fields 없음"
        if len(field_names) > 3:
            preview += ", ..."
        n_points = getattr(dataset, "n_points", "?")
        n_cells = getattr(dataset, "n_cells", "?")
        self._data_label.setText(
            f"데이터: 로드됨 ({n_points} pts, {n_cells} cells, {preview})"
        )
        self._run_btn.setText("실행 (로드 데이터)")

    def _refresh_op_list(self) -> None:
        """카테고리 필터에 맞춰 op 리스트 갱신."""
        cat = self._category_combo.currentText()
        self._op_list.clear()
        for op in self._facade.list_operations():
            info = self._facade.describe(op)
            all_label = self._tr("posttools.category.all")
            if cat in (all_label, "전체", "All") or info["category"] == cat:
                self._op_list.addItem(op)

    def _on_op_selected(self, op_name: str) -> None:
        if not op_name:
            return
        info = self._facade.describe(op_name)
        self._desc_label.setText(
            f"[{info['category']}] {info['description']}"
        )
        self._params_label.setText(", ".join(info["params"]))
        self._returns_label.setText(", ".join(info["returns"]))
        self._rebuild_param_form(op_name)
        self._refresh_preset_combo(op_name)

    def _refresh_preset_combo(self, op_name: str) -> None:
        """현재 op에 대한 프리셋 콤보 갱신."""
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        self._preset_combo.addItem(self._tr("posttools.preset.none"))
        for name in self._preset_store.list_presets(op_name):
            self._preset_combo.addItem(name)
        self._preset_combo.blockSignals(False)

    def _on_preset_selected(self, name: str) -> None:
        """프리셋 선택 시 해당 값을 폼에 적용."""
        none_label = self._tr("posttools.preset.none")
        if not name or name == none_label or name == "(없음)" or name == "(none)":
            return
        item = self._op_list.currentItem()
        if item is None:
            return
        op_name = item.text()
        preset = self._preset_store.get(op_name, name)
        if not preset:
            return
        for k, v in preset.items():
            widget = self._param_widgets.get(k)
            if widget is None:
                continue
            try:
                if isinstance(widget, QSpinBox):
                    widget.setValue(int(v))
                elif isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(v))
                elif isinstance(widget, QComboBox):
                    widget.setCurrentText(str(v))
            except (TypeError, ValueError):
                continue

    def add_user_preset(
        self, op_name: str, preset_name: str, params: dict[str, Any] | None = None,
    ) -> None:
        """현재 폼 값 (또는 명시 params)을 사용자 프리셋으로 저장.

        Args:
            op_name: 대상 op.
            preset_name: 저장 이름.
            params: 명시적 파라미터; None이면 현재 폼 값 사용.
        """
        if params is None:
            params = self._read_param_values()
        self._preset_store.add(op_name, preset_name, params)
        item = self._op_list.currentItem()
        if item is not None and item.text() == op_name:
            self._refresh_preset_combo(op_name)

    def _rebuild_param_form(self, op_name: str) -> None:
        """선택된 op의 scalar 파라미터에 맞춰 동적 폼 재생성."""
        # 기존 폼 비우기
        while self._param_form_layout.rowCount() > 0:
            self._param_form_layout.removeRow(0)
        self._param_widgets.clear()

        specs = self._facade.scalar_param_specs(op_name)
        if not specs:
            label = QLabel(self._tr("posttools.params.empty"))
            label.setStyleSheet("color: #888;")
            self._param_form_layout.addRow(label)
            return

        for name, spec in specs.items():
            widget = self._build_param_widget(spec)
            self._param_form_layout.addRow(f"{name}:", widget)
            self._param_widgets[name] = widget

    @staticmethod
    def _build_param_widget(spec: dict[str, Any]) -> QWidget:
        """spec dict → 적절한 입력 위젯."""
        ptype = spec.get("type", "str")
        default = spec.get("default")
        if ptype == "int":
            w = QSpinBox()
            w.setRange(int(spec.get("min", -10**9)), int(spec.get("max", 10**9)))
            w.setValue(int(default if default is not None else 0))
            return w
        if ptype == "float":
            w = QDoubleSpinBox()
            w.setDecimals(8)
            w.setRange(float(spec.get("min", -1e18)), float(spec.get("max", 1e18)))
            w.setValue(float(default if default is not None else 0.0))
            return w
        if ptype == "str":
            w = QComboBox()
            options = spec.get("options", [])
            for opt in options:
                w.addItem(str(opt))
            if default in options:
                w.setCurrentText(str(default))
            return w
        # fallback
        w = QLabel(str(default))
        return w

    def _read_param_values(self) -> dict[str, Any]:
        """현재 폼 위젯들의 값을 dict로 추출."""
        values: dict[str, Any] = {}
        for name, widget in self._param_widgets.items():
            if isinstance(widget, QSpinBox):
                values[name] = int(widget.value())
            elif isinstance(widget, QDoubleSpinBox):
                values[name] = float(widget.value())
            elif isinstance(widget, QComboBox):
                values[name] = widget.currentText()
        return values

    def _on_run_clicked(self) -> None:
        op_name = self._op_list.currentItem()
        if op_name is None:
            self._result_text.setPlainText("연산을 선택하세요.")
            return
        op_name = op_name.text()
        kwargs: dict[str, Any] = {}
        try:
            kwargs, source = self._build_run_kwargs(op_name)
            result = self._facade.run(op_name, **kwargs)
            source_label = "로드 데이터셋" if source == "dataset" else "합성 데모 데이터"
            self._result_text.setPlainText(
                f"입력: {source_label}\n{self._summarize_result(result)}"
            )
            # 차트 갱신
            if self._chart is not None and hasattr(self._chart, "render"):
                try:
                    self._chart.render(op_name, result)
                except Exception:  # noqa: BLE001
                    pass
            # Export 활성화 + 결과 캐싱
            self._last_result = result
            self._last_op_name = op_name
            self._export_csv_btn.setEnabled(True)
            self._export_json_btn.setEnabled(True)
            self._export_npz_btn.setEnabled(True)
            self._save_chart_btn.setEnabled(self._chart is not None)
            # 이력 기록
            self._history.record(op_name, kwargs, result, status="ok")
            self.operation_done.emit(op_name, result)
        except Exception as e:
            self._result_text.setPlainText(f"실행 실패: {e}")
            self._last_result = None
            self._last_op_name = None
            self._export_csv_btn.setEnabled(False)
            self._export_json_btn.setEnabled(False)
            self._export_npz_btn.setEnabled(False)
            self._save_chart_btn.setEnabled(False)
            # 실패도 이력 기록
            self._history.record(op_name, kwargs, None, status="error", error=str(e))
            if self._chart is not None and hasattr(self._chart, "clear"):
                try:
                    self._chart.clear()
                except Exception:  # noqa: BLE001
                    pass

    def _on_save_chart(self) -> None:
        if self._chart is None or not hasattr(self._chart, "save_figure"):
            return
        from PySide6.QtWidgets import QFileDialog

        default = f"{self._last_op_name or 'chart'}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "차트 이미지 저장", default,
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if not path:
            return
        try:
            self._chart.save_figure(path)
            self._result_text.append(f"\n[차트 저장] {path}")
        except Exception as e:  # noqa: BLE001
            self._result_text.append(f"\n[차트 저장 실패] {e}")

    def history(self) -> list[dict[str, Any]]:
        """실행 이력 반환 (외부 GUI/검증용)."""
        return self._history.entries()

    def _on_save_preset(self) -> None:
        """현재 op + 폼 값을 사용자 프리셋으로 저장."""
        item = self._op_list.currentItem()
        if item is None:
            return
        op_name = item.text()
        if not self._param_widgets:
            self._result_text.append(
                f"\n[프리셋] {op_name}은 scalar 파라미터 없음.",
            )
            return
        from PySide6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self, "프리셋 저장", f"{op_name}의 프리셋 이름:",
        )
        if not ok or not name.strip():
            return
        try:
            self.add_user_preset(op_name, name.strip())
            self._result_text.append(
                f"\n[프리셋] '{name}' 저장 완료 ({op_name}).",
            )
        except Exception as e:  # noqa: BLE001
            self._result_text.append(f"\n[프리셋 저장 실패] {e}")

    def _on_show_history(self) -> None:
        """실행 이력 다이얼로그를 띄운다."""
        try:
            from naviertwin.gui.widgets.history_dialog import HistoryDialog
        except Exception as e:  # noqa: BLE001
            self._result_text.append(f"\n[이력 viewer 로드 실패] {e}")
            return
        dlg = HistoryDialog(self.history(), parent=self)
        dlg.replay_requested.connect(self._replay_from_history)
        dlg.exec()

    def _replay_from_history(self, idx: int) -> None:
        """이력 항목의 op + 파라미터를 폼에 복원."""
        entries = self.history()
        if not (0 <= idx < len(entries)):
            return
        e = entries[idx]
        op_name = e.get("op")
        if not op_name:
            return
        # op 선택
        items = self._op_list.findItems(
            op_name,
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MatchFlag.MatchExactly,
        )
        if not items:
            return
        self._op_list.setCurrentItem(items[0])
        # 폼 값 적용
        kwargs = e.get("kwargs_summary") or {}
        for k, v in kwargs.items():
            widget = self._param_widgets.get(k)
            if widget is None:
                continue
            try:
                from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox

                if isinstance(widget, QSpinBox):
                    widget.setValue(int(v))
                elif isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(v))
                elif isinstance(widget, QComboBox):
                    widget.setCurrentText(str(v))
            except (TypeError, ValueError):
                continue
        self._result_text.append(
            f"\n[재실행 준비] 이력 #{idx} ({op_name}) 폼 복원 완료.",
        )

    def _on_export_csv(self) -> None:
        self._export_with("csv")

    def _on_export_json(self) -> None:
        self._export_with("json")

    def _on_export_npz(self) -> None:
        self._export_with("npz")

    def _export_with(self, fmt: str) -> None:
        if self._last_result is None or self._last_op_name is None:
            return
        from PySide6.QtWidgets import QFileDialog

        filters = {
            "csv": "CSV (*.csv)",
            "json": "JSON (*.json)",
            "npz": "NPZ (*.npz)",
        }
        default = f"{self._last_op_name}_result.{fmt}"
        path, _ = QFileDialog.getSaveFileName(
            self, "결과 저장", default, filters[fmt],
        )
        if not path:
            return
        try:
            from naviertwin.core.post_process_export import (
                save_csv,
                save_json,
                save_npz,
            )

            if fmt == "csv":
                save_csv(self._last_result, path)
            elif fmt == "json":
                save_json(self._last_result, path)
            else:
                save_npz(self._last_result, path)
            self._result_text.append(f"\n[저장] {fmt.upper()} → {path}")
        except Exception as e:  # noqa: BLE001
            self._result_text.append(f"\n[저장 실패] {e}")

    def _on_run_category(self) -> None:
        cat = self._category_combo.currentText()
        all_label = self._tr("posttools.category.all")
        if cat in (all_label, "전체", "All"):
            self._result_text.setPlainText(
                "'전체/All' 카테고리는 일괄 실행 비활성. 특정 카테고리를 선택하세요."
            )
            return
        try:
            from naviertwin.core.post_process_export import (
                bulk_summary_markdown,
                run_category,
            )

            # 모든 op의 smoke kwargs 수집
            all_smokes: dict[str, dict[str, Any]] = {}
            for op in self._facade.list_operations():
                try:
                    all_smokes[op] = self._build_smoke_kwargs(op)
                except Exception:  # noqa: BLE001
                    pass
            results = run_category(self._facade, cat, smoke_kwargs=all_smokes)
            self._result_text.setPlainText(bulk_summary_markdown(results))
            self._last_result = {"bulk": results}  # type: ignore[assignment]
            self._last_op_name = f"bulk_{cat}"
            self._export_csv_btn.setEnabled(False)  # bulk는 CSV 불가
            self._export_json_btn.setEnabled(True)
            self._export_npz_btn.setEnabled(False)
        except Exception as e:  # noqa: BLE001
            self._result_text.setPlainText(f"일괄 실행 실패: {e}")

    def _build_run_kwargs(self, op_name: str) -> tuple[dict[str, Any], str]:
        """현재 패널 상태에 맞는 op 입력을 구성한다.

        Scalar 파라미터는 사용자 폼 값으로 덮어쓴다 (배열은 dataset/smoke).
        """
        if self._dataset is not None:
            kwargs = self._build_dataset_kwargs(op_name, self._dataset)
            source = "dataset"
        else:
            kwargs = self._build_smoke_kwargs(op_name)
            source = "demo"
        # 사용자 입력 scalar 파라미터로 덮어쓰기
        kwargs.update(self._read_param_values())
        return kwargs, source

    @classmethod
    def _build_dataset_kwargs(cls, op_name: str, dataset: object) -> dict[str, Any]:
        """로드된 CFD dataset에서 facade op 입력을 자동 구성한다."""
        fields = cls._dataset_numeric_fields(dataset)
        signal = cls._primary_signal(fields)
        matrix = cls._field_matrix(fields)

        if op_name == "psd_welch":
            return {"signal": signal, "fs": 1.0, "nperseg": min(128, signal.size)}
        if op_name == "reynolds_stats":
            return cls._reynolds_kwargs(matrix)
        if op_name == "quadrant_analysis":
            return cls._quadrant_kwargs(matrix)
        if op_name == "kolmogorov_slope":
            return {"signal": signal, "dx": 1.0}
        if op_name == "box_stats":
            return {"x": signal, "whisker_factor": 1.5}
        if op_name == "anomaly_mahalanobis":
            return {"X": matrix}
        if op_name == "ts_features":
            return {"signal": signal}
        if op_name == "change_points":
            return {"signal": signal, "n_changepoints": 1, "method": "binary"}
        if op_name == "denoise":
            return cls._denoise_kwargs(signal)
        if op_name == "phase_average":
            period = max(2.0, float(signal.size // 4))
            return {"t": np.arange(signal.size, dtype=np.float64), "signal": signal,
                    "period": period, "n_bins": min(20, max(4, signal.size // 2))}
        if op_name == "eof":
            X = cls._snapshot_matrix(dataset, fields)
            return {"X": X, "n_modes": max(1, min(5, *X.shape))}
        if op_name == "safe_eval":
            variables = cls._safe_eval_variables(fields)
            expression = "sqrt(u**2 + v**2)" if "v" in variables else "u"
            return {"expression": expression, "variables": variables}
        if op_name == "two_point_acf":
            u = cls._two_point_matrix(dataset, fields)
            return {"u": u, "dx": 1.0, "max_lag": min(20, max(1, u.shape[1] - 1))}
        if op_name == "running_moments":
            return {"samples": matrix}
        if op_name == "pod_truncation":
            X = cls._snapshot_matrix(dataset, fields)
            singular_values = np.linalg.svd(X - X.mean(axis=0, keepdims=True), compute_uv=False)
            return {"singular_values": singular_values, "fraction": 0.99}
        if op_name == "quantile":
            return {"x": signal, "q": 50.0}
        if op_name == "time_interp":
            X = cls._snapshot_matrix(dataset, fields)
            times = np.arange(X.shape[0], dtype=np.float64)
            return {"snapshots": X, "times": times, "t_query": float(times.mean()),
                    "n_uniform": min(8, max(2, X.shape[0]))}
        if op_name == "coord_transform":
            xyz = cls._mesh_points(dataset)
            vectors = cls._vector_matrix(fields, xyz.shape[0])
            return {"xyz": xyz, "vectors": vectors}
        if op_name == "line_probe":
            xyz = cls._mesh_points(dataset)
            field = cls._field_for_points(signal, xyz.shape[0])
            return {"points": xyz, "field": field, "start": xyz.min(axis=0),
                    "end": xyz.max(axis=0), "n_samples": min(32, max(2, xyz.shape[0]))}
        if op_name == "gof_normality":
            return {"x": signal}
        if op_name == "conditional_sampling":
            return {"signal": signal, "threshold": float(np.median(signal)),
                    "half_window": min(8, max(1, signal.size // 8))}
        if op_name == "grid_derivatives":
            field_2d = cls._square_field(signal)
            return {"field_2d": field_2d, "dx": 1.0, "dy": 1.0}
        if op_name == "critical_points":
            u, v = cls._square_vector_components(fields)
            return {"u": u, "v": v, "dx": 1.0, "dy": 1.0}
        if op_name == "anisotropy_state":
            return {"reynolds_stress": cls._reynolds_stress(matrix)}
        if op_name == "morphology_components":
            field_2d = cls._square_field(signal)
            return {"field": field_2d, "threshold": float(np.median(field_2d)), "min_size": 4}

        raise ValueError(
            f"'{op_name}' 연산은 현재 데이터셋에서 필요한 입력을 자동 구성할 수 없습니다. "
            "데이터셋 없이 Demo 실행으로 합성 입력을 확인하세요."
        )

    @staticmethod
    def _build_smoke_kwargs(op_name: str) -> dict[str, Any]:
        """각 op에 대한 합리적인 smoke 데이터 생성."""
        rng = np.random.default_rng(0)
        base_signal = np.sin(np.linspace(0, 4 * np.pi, 500)) + 0.1 * rng.standard_normal(500)
        unit_square = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ], dtype=np.float64)
        xyz = rng.uniform(0.2, 1.2, (80, 3))
        grid_x = np.linspace(-1.0, 1.0, 32)
        grid_y = np.linspace(-1.0, 1.0, 32)
        X, Y = np.meshgrid(grid_x, grid_y, indexing="ij")
        kwargs_map: dict[str, dict[str, Any]] = {
            "psd_welch": {"signal": base_signal, "fs": 100.0, "nperseg": 128, "window": "hann"},
            "reynolds_stats": {"u": rng.standard_normal((100, 5))},
            "surface_forces": {
                "triangles": unit_square,
                "pressure": np.array([1.0, 1.0]),
                "shear_traction": np.tile([0.1, 0.0, 0.0], (2, 1)),
                "rho": 1.225,
                "u_inf": 10.0,
                "area_ref": 1.0,
            },
            "plane_flux": {
                "triangles": unit_square,
                "velocity": np.tile([0.0, 0.0, 1.0], (2, 1)),
                "scalar": np.array([300.0, 320.0]),
                "density": np.array([1.0, 1.1]),
            },
            "stat_convergence": {
                "signal": 1.0 + 0.05 * rng.standard_normal(2000),
                "n_batches": 20,
                "window": 100,
            },
            "quadrant_analysis": {
                "up": rng.standard_normal(2000),
                "vp": rng.standard_normal(2000),
                "hole": 0.0,
            },
            "kolmogorov_slope": {"signal": base_signal, "dx": 1.0},
            "box_stats": {"x": rng.standard_normal(500), "whisker_factor": 1.5},
            "anomaly_mahalanobis": {"X": rng.standard_normal((100, 3))},
            "ts_features": {"signal": base_signal},
            "change_points": {
                "signal": np.concatenate([
                    rng.standard_normal(60), 5 + rng.standard_normal(60),
                ]),
                "n_changepoints": 1,
                "method": "binary",
            },
            "denoise": {"signal": base_signal, "window_length": 21, "polyorder": 3},
            "phase_average": {
                "t": np.linspace(0, 50, 2000),
                "signal": np.sin(2 * np.pi * np.linspace(0, 50, 2000)),
                "period": 1.0,
                "n_bins": 20,
            },
            "eof": {"X": rng.standard_normal((100, 30)), "n_modes": 5},
            "safe_eval": {
                "expression": "sqrt(u**2 + v**2)",
                "variables": {"u": np.array([1.0, 2.0, 3.0]), "v": np.array([0.0, 1.0, 0.0])},
            },
            "two_point_acf": {"u": rng.standard_normal((100, 50)), "dx": 0.1, "max_lag": 20},
            "running_moments": {"samples": rng.standard_normal((100, 4))},
            "pod_truncation": {
                "singular_values": np.array([10.0, 5.0, 1.0, 0.1]),
                "fraction": 0.99,
            },
            "quantile": {"x": np.arange(101.0), "q": 50.0},
            "critical_points": {
                "u": -np.outer(np.linspace(-1, 1, 21), np.ones(21)).T,
                "v": np.outer(np.linspace(-1, 1, 21), np.ones(21)),
                "dx": 0.1,
                "dy": 0.1,
            },
            "time_interp": {
                "snapshots": np.stack([
                    np.array([t, t * t, np.sin(t)], dtype=np.float64)
                    for t in np.linspace(0.0, 1.0, 6)
                ]),
                "times": np.linspace(0.0, 1.0, 6),
                "t_query": 0.45,
                "n_uniform": 8,
            },
            "coord_transform": {
                "xyz": xyz[:12],
                "vectors": rng.standard_normal((12, 3)),
            },
            "line_probe": {
                "points": xyz,
                "field": xyz[:, 0] + 0.5 * xyz[:, 1],
                "start": np.array([0.2, 0.2, 0.2]),
                "end": np.array([1.2, 1.2, 1.2]),
                "n_samples": 16,
            },
            "gof_normality": {"x": rng.standard_normal(300)},
            "conditional_sampling": {
                "signal": base_signal,
                "threshold": 0.0,
                "half_window": 8,
            },
            "grid_derivatives": {
                "field_2d": X ** 2 + Y ** 2,
                "dx": grid_x[1] - grid_x[0],
                "dy": grid_y[1] - grid_y[0],
            },
            "anisotropy_state": {
                "reynolds_stress": np.diag([1.2, 0.8, 0.6]),
            },
            "morphology_components": {
                "field": ((X ** 2 + Y ** 2) < 0.45).astype(float),
                "threshold": 0.5,
                "min_size": 4,
            },
            "cell_volume_integrals": {
                "vertices": np.array([
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [2.0, 0.0, 1.0],
                ], dtype=np.float64),
                "connectivity": np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64),
                "field": np.array([2.0, 4.0]),
            },
            "helmholtz_decomp": {
                "u": rng.standard_normal((16, 16)),
                "v": rng.standard_normal((16, 16)),
            },
            "rom_residual": {
                "X": rng.standard_normal((20, 30)),
                "basis": np.linalg.qr(rng.standard_normal((30, 5)))[0],
            },
            "rom_envelope": {
                "coeffs_train": rng.standard_normal((100, 5)),
                "new_coeff": np.zeros(5),
            },
            "subspace_drift": {
                "basis_old": np.linalg.qr(rng.standard_normal((20, 4)))[0],
                "basis_new": np.linalg.qr(rng.standard_normal((20, 4)))[0],
            },
            "gappy_reconstruct": {
                "basis": np.linalg.qr(rng.standard_normal((30, 4)))[0],
                "partial": rng.standard_normal(30),
                "mask": np.array([True] * 20 + [False] * 10, dtype=bool),
            },
            "surrogate_metrics": {
                "y_true": rng.standard_normal(100),
                "y_pred": rng.standard_normal(100),
            },
            "residual_diagnostics": {
                "residuals": rng.standard_normal(500),
            },
            "ensemble_average": {
                "predictions": [rng.standard_normal(10), rng.standard_normal(10)],
                "weights": np.array([0.6, 0.4]),
            },
            "trajectory_clustering": {
                "coeffs": rng.standard_normal((100, 5)),
                "window": 20,
                "n_clusters": 3,
            },
            "acoustic_strouhal": {
                "f": 100.0,
                "L": 0.1,
                "U": 10.0,
            },
            "save_rom": {
                "path": "/tmp/_facade_rom_smoke.npz",
                "modes": rng.standard_normal((20, 5)),
                "singular_values": np.array([5.0, 3.0, 2.0, 1.0, 0.5]),
            },
            "load_rom": {
                "path": "/tmp/_facade_rom_smoke.npz",
            },
            "mode_summary": {
                "spatial_modes": np.linalg.qr(rng.standard_normal((20, 4)))[0],
                "singular_values": np.array([5.0, 3.0, 2.0, 1.0]),
                "temporal_modes": rng.standard_normal((4, 30)),
            },
            "basis_interpolate": {
                "bases": [
                    np.linalg.qr(rng.standard_normal((20, 3)))[0],
                    np.linalg.qr(rng.standard_normal((20, 3)))[0],
                ],
                "params": np.array([0.0, 1.0]),
                "target": 0.5,
            },
            "batch_predict": {
                "predict_fn": lambda X: np.sum(X ** 2, axis=1),
                "X": rng.standard_normal((100, 5)),
                "chunk_size": 25,
            },
            "morris_sensitivity": {
                "f": lambda X: X[:, 0] + 2 * X[:, 1],
                "bounds": np.array([[0.0, 1.0]] * 3),
                "n_trajectories": 5,
                "n_levels": 4,
            },
            "permutation_importance": {
                "f": lambda X: 5 * X[:, 0] + 0.1 * X[:, 1],
                "X": rng.uniform(0, 1, (100, 2)),
                "y": rng.uniform(0, 1, 100),
                "n_repeats": 3,
            },
            "bic_model_average": {
                "predictions": [rng.standard_normal(10), rng.standard_normal(10)],
                "bic_values": np.array([100.0, 105.0]),
            },
            "stacking": {
                "predictions": rng.standard_normal((50, 3)),
                "y_true": rng.standard_normal(50),
                "nonnegative": True,
            },
            "mass_search": {
                "query": base_signal[100:130].copy(),
                "series": base_signal,
            },
            "find_motifs": {
                "series": base_signal,
                "window": 30,
                "k": 1,
            },
            "auto_report_probe": {
                "signal": base_signal,
                "fs": 100.0,
                "period_hint": None,
            },
            "auto_report_field": {
                "X": rng.standard_normal((100, 30)),
                "n_modes": 5,
            },
        }
        if op_name not in kwargs_map:
            raise ValueError(f"smoke 데이터 미정의: {op_name}")
        return kwargs_map[op_name]

    @staticmethod
    def _dataset_numeric_fields(dataset: object) -> dict[str, np.ndarray]:
        """CFDDataset mesh에서 숫자 point/cell field를 추출한다."""
        mesh = getattr(dataset, "mesh", None)
        if mesh is None:
            raise ValueError("dataset.mesh가 없습니다.")

        fields: dict[str, np.ndarray] = {}
        for store_name in ("point_data", "cell_data"):
            store = getattr(mesh, store_name, None)
            if not hasattr(store, "items"):
                continue
            for name, value in store.items():
                try:
                    arr = np.asarray(value, dtype=np.float64)
                except (TypeError, ValueError):
                    continue
                if arr.size == 0:
                    continue
                fields[str(name)] = np.nan_to_num(arr, copy=False)

        if not fields:
            raise ValueError("로드된 데이터셋에 숫자 필드가 없습니다.")
        return fields

    @classmethod
    def _primary_signal(cls, fields: dict[str, np.ndarray]) -> np.ndarray:
        """첫 번째 숫자 field를 1D signal로 변환한다."""
        for arr in fields.values():
            signal = cls._to_scalar_series(arr)
            if signal.size >= 2:
                return signal
        raise ValueError("후처리에 사용할 1D 숫자 필드가 없습니다.")

    @staticmethod
    def _to_scalar_series(arr: np.ndarray) -> np.ndarray:
        """스칼라/벡터 field를 1D 스칼라 시리즈로 정규화한다."""
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            return arr.reshape(-1)
        if arr.ndim >= 2:
            return np.linalg.norm(arr, axis=-1).reshape(-1)
        return arr.reshape(-1)

    @classmethod
    def _field_matrix(cls, fields: dict[str, np.ndarray]) -> np.ndarray:
        """여러 숫자 field를 공통 길이 행렬로 결합한다."""
        columns: list[np.ndarray] = []
        target_len: int | None = None
        for arr in fields.values():
            arr = np.asarray(arr, dtype=np.float64)
            if arr.ndim == 2 and 1 < arr.shape[1] <= 4:
                candidate_columns = [arr[:, i].reshape(-1) for i in range(arr.shape[1])]
            else:
                candidate_columns = [cls._to_scalar_series(arr)]
            for col in candidate_columns:
                if target_len is None:
                    target_len = col.size
                if col.size == target_len:
                    columns.append(col)
        if not columns:
            raise ValueError("행렬로 결합할 숫자 필드가 없습니다.")
        return np.column_stack(columns)

    @staticmethod
    def _reynolds_kwargs(matrix: np.ndarray) -> dict[str, np.ndarray]:
        kwargs = {"u": matrix[:, 0]}
        if matrix.shape[1] >= 2:
            kwargs["v"] = matrix[:, 1]
        if matrix.shape[1] >= 3:
            kwargs["w"] = matrix[:, 2]
        return kwargs

    @staticmethod
    def _quadrant_kwargs(matrix: np.ndarray) -> dict[str, np.ndarray | float]:
        if matrix.shape[1] < 2:
            raise ValueError("quadrant_analysis에는 최소 두 개의 숫자 필드가 필요합니다.")
        return {
            "up": matrix[:, 0] - matrix[:, 0].mean(),
            "vp": matrix[:, 1] - matrix[:, 1].mean(),
            "hole": 0.0,
        }

    @staticmethod
    def _denoise_kwargs(signal: np.ndarray) -> dict[str, Any]:
        window = min(21, signal.size if signal.size % 2 == 1 else signal.size - 1)
        window = max(5, window)
        if window >= signal.size:
            window = signal.size - 1 if signal.size % 2 == 0 else signal.size
        if window < 5:
            raise ValueError("denoise에는 최소 5개 이상의 값이 필요합니다.")
        return {"signal": signal, "window_length": window, "polyorder": min(3, window - 2)}

    @staticmethod
    def _safe_eval_variables(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        signals = [PostProcessPanel._to_scalar_series(arr) for arr in fields.values()]
        variables = {"u": signals[0]}
        if len(signals) > 1 and signals[1].shape == signals[0].shape:
            variables["v"] = signals[1]
        return variables

    @classmethod
    def _snapshot_matrix(cls, dataset: object, fields: dict[str, np.ndarray]) -> np.ndarray:
        field_name = next(iter(fields))
        extractor = getattr(dataset, "extract_field_snapshots", None)
        if callable(extractor):
            try:
                snapshots = np.asarray(extractor(field_name), dtype=np.float64)
                if snapshots.ndim == 2 and min(snapshots.shape) >= 1:
                    X = snapshots.T
                    if X.shape[0] >= 2:
                        return X
            except Exception:  # noqa: BLE001 - fallback to visible field data.
                pass
        signal = cls._primary_signal(fields)
        return np.vstack([signal, np.roll(signal, 1)])

    @classmethod
    def _two_point_matrix(cls, dataset: object, fields: dict[str, np.ndarray]) -> np.ndarray:
        X = cls._snapshot_matrix(dataset, fields)
        if X.shape[1] >= 2:
            return X
        signal = cls._primary_signal(fields)
        return np.vstack([signal, np.roll(signal, 1)])

    @staticmethod
    def _mesh_points(dataset: object) -> np.ndarray:
        mesh = getattr(dataset, "mesh", None)
        points = np.asarray(getattr(mesh, "points", []), dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 2:
            raise ValueError("dataset mesh에 2개 이상의 점 좌표가 필요합니다.")
        if points.shape[1] == 2:
            points = np.column_stack([points, np.zeros(points.shape[0])])
        if points.shape[1] != 3:
            raise ValueError(f"mesh points shape must be (N, 3), got {points.shape}")
        return points

    @classmethod
    def _vector_matrix(cls, fields: dict[str, np.ndarray], n_points: int) -> np.ndarray:
        for arr in fields.values():
            arr = np.asarray(arr, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] == n_points and arr.shape[1] >= 3:
                return arr[:, :3]
        signal = cls._primary_signal(fields)
        signal = cls._field_for_points(signal, n_points)
        return np.column_stack([signal, np.zeros(n_points), np.zeros(n_points)])

    @staticmethod
    def _field_for_points(signal: np.ndarray, n_points: int) -> np.ndarray:
        if signal.size == n_points:
            return signal
        if signal.size > n_points:
            return signal[:n_points]
        raise ValueError(
            f"필드 길이({signal.size})가 mesh points({n_points})보다 짧습니다."
        )

    @staticmethod
    def _square_field(signal: np.ndarray) -> np.ndarray:
        n = int(np.sqrt(signal.size))
        if n * n != signal.size or n < 2:
            raise ValueError("이 연산에는 정사각 2D 격자로 재구성 가능한 필드가 필요합니다.")
        return signal[: n * n].reshape(n, n)

    @classmethod
    def _square_vector_components(
        cls, fields: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        matrix = cls._field_matrix(fields)
        if matrix.shape[1] < 2:
            raise ValueError("2D 벡터장 성분 두 개가 필요합니다.")
        return cls._square_field(matrix[:, 0]), cls._square_field(matrix[:, 1])

    @staticmethod
    def _reynolds_stress(matrix: np.ndarray) -> np.ndarray:
        if matrix.shape[1] < 3:
            padded = np.zeros((matrix.shape[0], 3), dtype=np.float64)
            padded[:, : matrix.shape[1]] = matrix
            matrix = padded
        cov = np.cov(matrix[:, :3], rowvar=False)
        return np.asarray(cov, dtype=np.float64).reshape(3, 3)

    @staticmethod
    def _summarize_result(result: dict[str, Any]) -> str:
        """결과 dict를 사람이 읽기 쉬운 텍스트로 요약."""
        lines = []
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                if v.size <= 6:
                    lines.append(f"{k}: {v.tolist()}")
                else:
                    lines.append(
                        f"{k}: shape={v.shape}, mean={float(v.mean()):.4g}, "
                        f"std={float(v.std()):.4g}"
                    )
            elif isinstance(v, (int, float)):
                lines.append(f"{k}: {v}")
            elif isinstance(v, list):
                if len(v) <= 6:
                    lines.append(f"{k}: {v}")
                else:
                    lines.append(f"{k}: list len={len(v)}")
            elif isinstance(v, dict):
                lines.append(f"{k}: dict keys={list(v.keys())[:5]}")
            else:
                lines.append(f"{k}: {type(v).__name__}")
        return "\n".join(lines)


__all__ = ["PostProcessPanel"]
