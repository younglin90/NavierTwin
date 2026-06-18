"""해석해 ↔ 수치해 비교 시각화 위젯.

Matplotlib 선 그래프 + 메트릭 테이블을 QWidget 에 임베드한다.
"""

from __future__ import annotations

from collections import deque
from functools import partial
from typing import Any

from PySide6.QtWidgets import (
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_METRIC_KEYS = ("rmse", "r2", "relative_l2", "max_error")


def _init_metric_row(table: QTableWidget, entry: tuple[int, str]) -> None:
    row, key = entry
    table.setItem(row, 0, QTableWidgetItem(key))
    table.setItem(row, 1, QTableWidgetItem("—"))


def _update_metric_row(
    table: QTableWidget, metrics: dict[str, Any], entry: tuple[int, str]
) -> None:
    row, key = entry
    val = metrics.get(key)
    text = f"{val:.6g}" if isinstance(val, (int, float)) else "—"
    table.setItem(row, 1, QTableWidgetItem(text))


class AnalyticCompareWidget(QWidget):
    """해석해 vs 수치해 비교 시각화 위젯.

    - 상단: matplotlib 선 그래프 (해석해 + 수치해)
    - 하단: 메트릭 테이블 (RMSE / R² / rel.L2 / max_error)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # matplotlib 임베드 (지연 import — matplotlib 없는 환경에서도 import 가능)
        try:
            from matplotlib.backends.backend_qtagg import (  # type: ignore[import-not-found]
                FigureCanvasQTAgg,
            )
            from matplotlib.figure import Figure  # type: ignore[import-not-found]

            self._figure = Figure(figsize=(5, 3.2), tight_layout=True)
            self._canvas = FigureCanvasQTAgg(self._figure)
            layout.addWidget(self._canvas, stretch=4)
            self._mpl_available = True
        except ImportError:
            from PySide6.QtWidgets import QLabel

            fallback = QLabel("matplotlib 필요 — pip install matplotlib")
            layout.addWidget(fallback, stretch=4)
            self._figure = None
            self._canvas = None
            self._mpl_available = False

        # 메트릭 테이블
        self._table = QTableWidget(4, 2, self)
        self._table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.setMaximumHeight(150)
        deque(map(partial(_init_metric_row, self._table), enumerate(_METRIC_KEYS)), maxlen=0)
        layout.addWidget(self._table, stretch=1)

    def update_result(self, result: dict[str, Any]) -> None:
        """비교 결과를 표시한다.

        Args:
            result: ``compare_against_analytic`` 반환 dict.
                - "metrics": dict
                - "analytic": ndarray
                - "numeric": ndarray
                - "coords": ndarray
        """
        metrics = result.get("metrics", {})
        update_row = partial(_update_metric_row, self._table, metrics)
        deque(map(update_row, enumerate(_METRIC_KEYS)), maxlen=0)

        if not self._mpl_available or self._figure is None:
            return

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        coords = result.get("coords")
        analytic = result.get("analytic")
        numeric = result.get("numeric")
        if coords is not None and analytic is not None and numeric is not None:
            ax.plot(coords, analytic, "k-", label="Analytic", linewidth=2)
            ax.plot(coords, numeric, "r.--", label="Numeric", markersize=4)
            ax.set_xlabel("coord")
            ax.set_ylabel("velocity")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
        if self._canvas is not None:
            self._canvas.draw_idle()
