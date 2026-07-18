"""QComboBox popup close regression tests."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def test_combo_close_fix_selects_and_hides_after_mouse_release(qtbot) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication, QComboBox

    from naviertwin.gui.utils.combo_fix import apply_combo_close_fix

    class RecordingCombo(QComboBox):
        def __init__(self) -> None:
            super().__init__()
            self.hide_count = 0

        def hidePopup(self) -> None:  # noqa: N802
            self.hide_count += 1
            super().hidePopup()

    combo = RecordingCombo()
    combo.addItems(["pressure", "velocity"])
    qtbot.addWidget(combo)

    apply_combo_close_fix(combo)
    index = combo.model().index(1, 0)
    rect = combo.view().visualRect(index)
    press = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        rect.center(),
        combo.view().viewport().mapToGlobal(rect.center()),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    release = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        rect.center(),
        combo.view().viewport().mapToGlobal(rect.center()),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(combo.view().viewport(), press)
    QApplication.sendEvent(combo.view().viewport(), release)

    qtbot.waitUntil(lambda: combo.hide_count >= 1, timeout=1000)
    assert combo.currentText() == "velocity"


def test_combo_close_fix_does_not_close_on_current_index_change(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QComboBox

    from naviertwin.gui.utils import combo_fix

    class RecordingCombo(QComboBox):
        def __init__(self) -> None:
            super().__init__()
            self.hide_count = 0

        def hidePopup(self) -> None:  # noqa: N802
            self.hide_count += 1
            super().hidePopup()

    combo = RecordingCombo()
    combo.addItems(["pressure", "velocity", "temperature"])
    qtbot.addWidget(combo)

    monkeypatch.setattr(combo_fix, "_hide_popup_later", lambda combo: combo.hidePopup())
    combo_fix.apply_combo_close_fix(combo)
    combo.setCurrentIndex(2)

    assert combo.currentText() == "temperature"
    assert combo.hide_count == 0


def test_combo_body_open_click_release_does_not_select_and_close(qtbot) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication, QComboBox

    from naviertwin.gui.utils.combo_fix import apply_combo_close_fix

    class RecordingCombo(QComboBox):
        def __init__(self) -> None:
            super().__init__()
            self.hide_count = 0

        def hidePopup(self) -> None:  # noqa: N802
            self.hide_count += 1
            super().hidePopup()

    combo = RecordingCombo()
    combo.addItems(["pressure", "velocity"])
    qtbot.addWidget(combo)
    combo.show()

    apply_combo_close_fix(combo)
    combo.showPopup()
    index = combo.model().index(1, 0)
    rect = combo.view().visualRect(index)

    combo_press = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        combo.rect().center(),
        combo.mapToGlobal(combo.rect().center()),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    popup_release = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        rect.center(),
        combo.view().viewport().mapToGlobal(rect.center()),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )

    QApplication.sendEvent(combo, combo_press)
    QApplication.sendEvent(combo.view().viewport(), popup_release)

    assert combo.currentText() == "pressure"
    assert combo.hide_count == 0


def test_main_window_installs_combo_close_fix_for_existing_controls(qtbot) -> None:
    from PySide6.QtWidgets import QComboBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    combos = win.findChildren(QComboBox)
    assert combos
    assert all(getattr(combo, "_naviertwin_close_fix", False) for combo in combos)


def test_close_confirmation_dialog_centers_on_main_window(qtbot) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    win.resize(900, 640)
    win.move(120, 90)
    win.show()

    dialog = win._build_close_confirmation_dialog()
    qtbot.addWidget(dialog)
    dialog.show()
    win._center_child_window(dialog)

    parent_center = win.frameGeometry().center()
    dialog_center = dialog.frameGeometry().center()
    assert abs(parent_center.x() - dialog_center.x()) <= 8
    assert abs(parent_center.y() - dialog_center.y()) <= 8
    dialog.close()
