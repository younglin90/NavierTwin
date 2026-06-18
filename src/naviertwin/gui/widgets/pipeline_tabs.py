"""Multi-row pipeline tab widget used by the main workflow."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class PipelineTabWidget(QWidget):
    """Small QTabWidget-compatible wrapper with a multi-row button tab bar."""

    currentChanged = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None, *, max_columns: int = 5) -> None:
        super().__init__(parent)
        self._max_columns = max(1, max_columns)
        self._widgets: list[QWidget] = []
        self._titles: list[str] = []
        self._buttons: list[QPushButton] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tab_bar = QWidget(self)
        self._tab_bar.setObjectName("pipelineTabBar")
        self._tab_layout = QGridLayout(self._tab_bar)
        self._tab_layout.setContentsMargins(8, 8, 8, 6)
        self._tab_layout.setHorizontalSpacing(8)
        self._tab_layout.setVerticalSpacing(6)
        layout.addWidget(self._tab_bar)

        self._stack = QStackedWidget(self)
        self._stack.currentChanged.connect(self._on_stack_changed)
        layout.addWidget(self._stack, stretch=1)

    def addTab(self, widget: QWidget, title: str) -> int:
        """Add a tab and return its index."""
        index = len(self._widgets)
        self._widgets.append(widget)
        self._titles.append(title)
        self._stack.addWidget(widget)

        button = QPushButton(title, self._tab_bar)
        button.setObjectName("pipelineTabButton")
        button.setCheckable(True)
        button.setMinimumHeight(38)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button.setToolTip(title)
        button.clicked.connect(lambda _checked=False, i=index: self.setCurrentIndex(i))
        self._buttons.append(button)
        self._reflow_buttons()
        self._update_button_state()
        return index

    def count(self) -> int:
        return len(self._widgets)

    def currentIndex(self) -> int:
        return self._stack.currentIndex()

    def currentWidget(self) -> QWidget | None:
        return self._stack.currentWidget()

    def indexOf(self, widget: QWidget | None) -> int:
        if widget is None:
            return -1
        try:
            return self._widgets.index(widget)
        except ValueError:
            return -1

    def setCurrentIndex(self, index: int) -> None:
        if 0 <= index < self.count():
            self._stack.setCurrentIndex(index)

    def setCurrentWidget(self, widget: QWidget) -> None:
        index = self.indexOf(widget)
        if index >= 0:
            self.setCurrentIndex(index)

    def tabText(self, index: int) -> str:
        if 0 <= index < self.count():
            return self._titles[index]
        return ""

    def setTabText(self, index: int, title: str) -> None:
        if not 0 <= index < self.count():
            return
        self._titles[index] = title
        button = self._buttons[index]
        button.setText(title)
        button.setToolTip(title)

    def widget(self, index: int) -> QWidget | None:
        if 0 <= index < self.count():
            return self._widgets[index]
        return None

    def setDocumentMode(self, _enabled: bool) -> None:
        """Compatibility no-op with QTabWidget call sites."""

    def setElideMode(self, _mode: Qt.TextElideMode) -> None:
        """Compatibility no-op; pipeline buttons show full labels."""

    def setTabPosition(self, _position: object) -> None:
        """Compatibility no-op; this widget always renders tabs above content."""

    def setUsesScrollButtons(self, _enabled: bool) -> None:
        """Compatibility no-op; multi-row buttons avoid scroll arrows."""

    def _on_stack_changed(self, index: int) -> None:
        self._update_button_state()
        self.currentChanged.emit(index)

    def _reflow_buttons(self) -> None:
        while self._tab_layout.count():
            item = self._tab_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(self._tab_bar)

        columns = min(self._max_columns, max(1, len(self._buttons)))
        index = 0
        while index < len(self._buttons):
            button = self._buttons[index]
            row = index // columns
            col = index % columns
            self._tab_layout.addWidget(button, row, col)
            index += 1
        col = 0
        while col < columns:
            self._tab_layout.setColumnStretch(col, 1)
            col += 1

    def _update_button_state(self) -> None:
        current = self.currentIndex()
        index = 0
        while index < len(self._buttons):
            button = self._buttons[index]
            button.setChecked(index == current)
            button.setProperty("active", index == current)
            button.style().unpolish(button)
            button.style().polish(button)
            index += 1


__all__ = ["PipelineTabWidget"]
