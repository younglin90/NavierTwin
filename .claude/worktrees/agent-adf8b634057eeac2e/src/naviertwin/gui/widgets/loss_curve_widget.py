"""학습 loss curve 실시간 시각화 위젯.

모델이 학습 중 epoch 마다 loss 를 append 하면 이 위젯이 plot 을 업데이트.
"""

from __future__ import annotations

from collections import deque
from functools import partial

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


def _copy_loss_series(item: tuple[str, list[float]]) -> tuple[str, list[float]]:
    label, losses = item
    return label, list(losses)


def _plot_loss_series(ax: object, item: tuple[str, list[float]]) -> None:
    label, losses = item
    if not losses:
        return
    ax.plot(range(1, len(losses) + 1), losses, label=label, linewidth=1.5)


class LossCurveWidget(QWidget):
    """Matplotlib 선 그래프 기반 loss curve 위젯."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: dict[str, list[float]] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg,
            )
            from matplotlib.figure import Figure

            self._figure = Figure(figsize=(5, 2.8), tight_layout=True)
            self._canvas = FigureCanvasQTAgg(self._figure)
            layout.addWidget(self._canvas)
            self._mpl = True
        except ImportError:
            layout.addWidget(QLabel("matplotlib 필요"))
            self._figure = None
            self._canvas = None
            self._mpl = False

    def set_losses(
        self, series: dict[str, list[float]], log_scale: bool = True
    ) -> None:
        """여러 loss 계열을 한 번에 그린다.

        Args:
            series: {label: [loss_0, loss_1, ...]}.
            log_scale: y 축 로그 여부.
        """
        self._series = dict(map(_copy_loss_series, series.items()))
        if not self._mpl or self._figure is None:
            return
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        deque(map(partial(_plot_loss_series, ax), self._series.items()), maxlen=0)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        if log_scale:
            try:
                ax.set_yscale("log")
            except ValueError:
                ax.set_yscale("linear")
        ax.grid(True, alpha=0.3)
        if self._series:
            ax.legend(loc="best", fontsize=8)
        if self._canvas is not None:
            self._canvas.draw_idle()


__all__ = ["LossCurveWidget"]
