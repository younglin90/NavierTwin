"""모델 비교 대시보드 위젯 — Matplotlib 바 차트로 RMSE/R² 비교.

여러 모델의 메트릭을 딕셔너리로 주입하면 비교 플롯을 갱신한다.

Usage:
    widget.update({
        "Kriging": {"rmse": 0.012, "r2": 0.98},
        "RBF": {"rmse": 0.018, "r2": 0.96},
        "FNO1D": {"rmse": 0.008, "r2": 0.99},
    })
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ModelCompareWidget(QWidget):
    """모델 메트릭 비교 — matplotlib 바 차트 + 테이블."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg,
            )
            from matplotlib.figure import Figure

            self._figure = Figure(figsize=(6, 3), tight_layout=True)
            self._canvas = FigureCanvasQTAgg(self._figure)
            layout.addWidget(self._canvas, stretch=4)
            self._mpl = True
        except ImportError:
            layout.addWidget(QLabel("matplotlib 필요"), stretch=4)
            self._figure = None
            self._canvas = None
            self._mpl = False

        self._table = QTableWidget(0, 3, self)
        self._table.setHorizontalHeaderLabels(["Model", "RMSE", "R²"])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table, stretch=1)

    def update(self, results: dict[str, dict[str, float]]) -> None:
        """모델 → 메트릭 dict 를 받아 대시보드 업데이트."""
        names = list(results.keys())
        rmses = [results[n].get("rmse", 0.0) for n in names]
        r2s = [results[n].get("r2", 0.0) for n in names]

        # 테이블
        self._table.setRowCount(len(names))
        for i, n in enumerate(names):
            self._table.setItem(i, 0, QTableWidgetItem(n))
            self._table.setItem(i, 1, QTableWidgetItem(f"{rmses[i]:.6g}"))
            self._table.setItem(i, 2, QTableWidgetItem(f"{r2s[i]:.6g}"))

        # 차트
        if not self._mpl or self._figure is None:
            return
        self._figure.clear()
        ax1 = self._figure.add_subplot(121)
        ax2 = self._figure.add_subplot(122)
        ax1.bar(names, rmses, color="tab:red")
        ax1.set_title("RMSE ↓")
        ax1.set_ylabel("RMSE")
        for label in ax1.get_xticklabels():
            label.set_rotation(30)
        ax2.bar(names, r2s, color="tab:green")
        ax2.set_title("R² ↑")
        ax2.set_ylim(0, 1)
        for label in ax2.get_xticklabels():
            label.set_rotation(30)
        if self._canvas is not None:
            self._canvas.draw_idle()


__all__ = ["ModelCompareWidget"]
