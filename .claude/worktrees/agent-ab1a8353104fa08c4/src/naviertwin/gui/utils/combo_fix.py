"""QComboBox 드롭다운 닫힘 버그 패치.

QSS 가 ``QComboBox QAbstractItemView`` 를 스타일링할 때 Qt6 의 기본 popup
종료 흐름이 끊겨 ``activated`` 시그널만으로는 닫히지 않는다. 가장 안정적인
해결은 popup view 의 mouse release 시점에 선택을 명시적으로 확정한 뒤
``hidePopup`` 을 호출하는 것이다.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QStyledItemDelegate,
    QWidget,
)


def apply_combo_close_fix(combo: QComboBox) -> None:
    """단일 QComboBox 에 닫힘 보정을 적용한다.

    동일 콤보에 두 번 적용되어도 안전하다 (idempotent flag).

    QSS 로 ``QComboBox QAbstractItemView`` 가 스타일링되면 popup 의 자동 종료
    흐름이 끊긴다. 다음 3중 보정으로 모든 케이스를 닫는다:

    1. ``QStyledItemDelegate`` 를 view 에 명시적으로 부여 (paint 정상화).
    2. ``view.clicked`` → 선택 확정 + 다음 프레임 ``hidePopup``.
    3. ``view.viewport`` 마우스 release 이벤트 필터 → 선택 확정 + ``hidePopup``.
    4. ``combo.activated`` → ``combo.hidePopup`` (키보드/엔터 보강).

    ``currentIndexChanged`` 에는 닫힘을 걸지 않는다. popup 이 열릴 때
    스타일/QPA 조합에 따라 highlight/current 변경이 먼저 발생할 수 있고, 그
    시점에 닫으면 "눌렀는데 안 열리는" 증상으로 보인다.

    또한 combo 본체 press 로 popup 이 열린 직후의 release 는 선택으로 보지
    않는다. 일부 QPA는 본체 가운데를 클릭해 popup 이 열리면 같은 클릭의
    release 를 popup view 로 전달해 즉시 선택/닫힘을 발생시킨다.
    """
    if getattr(combo, "_naviertwin_close_fix", False):
        return
    setattr(combo, "_naviertwin_close_fix", True)

    view = combo.view()
    if view is not None:
        try:
            view.setItemDelegate(QStyledItemDelegate(view))
        except Exception:
            pass

        # pressed 단계에서 popup 을 닫으면 Qt 가 아직 currentIndex 를 확정하기
        # 전이라 "선택도 안 되고 목록도 안 닫히는" 상태가 될 수 있다. clicked
        # 또는 mouse release 에서 명시적으로 setCurrentIndex 후 닫는다.
        try:
            view.clicked.connect(
                lambda index=None, combo=combo: _select_index_and_hide(combo, index)
            )
        except Exception:
            pass

        # 보강 — 일부 스타일은 pressed 가 emit 되지 않을 수 있어 viewport 의
        # MouseButtonRelease 도 후킹.
        _ViewClickFilter.attach(combo, view)

    # 키보드 Enter/Return 시 발생하는 activated 도 닫기.
    try:
        combo.activated.connect(
            lambda _index=None, combo=combo: _hide_popup_later(combo)
        )
    except Exception:
        pass

def _hide_popup_later(combo: QComboBox) -> None:
    """선택 이벤트 처리가 끝난 다음 프레임에 popup 을 강제로 닫는다."""

    def hide() -> None:
        try:
            combo.hidePopup()
        except RuntimeError:
            return
        except Exception:
            pass
        # 일부 스타일/QPA 조합에서 hidePopup 후에도 popup window 가 남는다.
        # 메인 윈도우가 아닌 popup container 만 직접 숨긴다.
        try:
            view = combo.view()
            popup = view.window() if view is not None else None
            if popup is not None and popup is not combo.window():
                popup.hide()
        except Exception:
            pass

    QTimer.singleShot(0, hide)


def _select_index_and_hide(combo: QComboBox, index: object) -> None:
    """Popup item index 를 현재 combo 값으로 확정하고 popup 을 닫는다."""
    try:
        if index is not None and index.isValid():  # type: ignore[attr-defined]
            combo.setCurrentIndex(int(index.row()))  # type: ignore[attr-defined]
            try:
                combo.activated.emit(int(index.row()))  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass
    _hide_popup_later(combo)


class _ViewClickFilter(QObject):
    """QComboBox view 의 mouse release 에서 선택 확정과 닫힘만 예약한다."""

    @classmethod
    def attach(cls, combo: QComboBox, view: QObject) -> None:
        flt = cls(combo)
        flt._combo = combo
        flt._pressed_in_view = False
        flt._ignore_release_from_combo_open = False
        existing = getattr(combo, "_naviertwin_view_filters", [])
        existing.append(flt)
        setattr(combo, "_naviertwin_view_filters", existing)
        combo.installEventFilter(flt)
        viewport = getattr(view, "viewport", lambda: view)()
        if viewport is not None:
            viewport.installEventFilter(flt)
        view.installEventFilter(flt)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: D401
        combo = getattr(self, "_combo", None)
        if combo is not None and watched is combo:
            if event.type() == QEvent.Type.MouseButtonPress:
                self._ignore_release_from_combo_open = True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                # release 가 combo 로 돌아왔으면 popup view 로 전달되지 않은
                # 정상 경로다. 다음 클릭은 평소처럼 처리한다.
                self._ignore_release_from_combo_open = False
            return False

        if event.type() == QEvent.Type.MouseButtonPress:
            self._pressed_in_view = True
            self._ignore_release_from_combo_open = False
            return False
        if event.type() == QEvent.Type.MouseButtonRelease:
            if (
                getattr(self, "_ignore_release_from_combo_open", False)
                and not getattr(self, "_pressed_in_view", False)
            ):
                self._ignore_release_from_combo_open = False
                event.accept()  # type: ignore[attr-defined]
                return True
            if not getattr(self, "_pressed_in_view", False):
                return False
            self._pressed_in_view = False
            if combo is not None:
                view = combo.view()
                index = None
                try:
                    viewport = view.viewport()
                    if watched is viewport:
                        pos = event.position().toPoint()  # type: ignore[attr-defined]
                    else:
                        global_pos = event.globalPosition().toPoint()  # type: ignore[attr-defined]
                        pos = viewport.mapFromGlobal(global_pos)
                    index = view.indexAt(pos)
                except Exception:
                    index = None
                if index is not None and index.isValid():
                    _select_index_and_hide(combo, index)
                # Qt 기본 popup 선택/hover 흐름을 막지 않는다. 이전 구현처럼
                # 이벤트를 소비하면 일부 QPA/스타일에서 목록 열림과 선택이
                # 불안정해진다.
                return False
        return False


def apply_to_widget_tree(root: QWidget) -> int:
    """``root`` 내부의 모든 QComboBox 에 닫힘 보정을 적용한다.

    Returns:
        보정한 콤보 개수.
    """
    return sum(map(_apply_combo_close_fix_if_needed, root.findChildren(QComboBox)))


def _apply_combo_close_fix_if_needed(combo: QComboBox) -> int:
    if getattr(combo, "_naviertwin_close_fix", False):
        return 0
    apply_combo_close_fix(combo)
    return 1


class _ComboCloseFilter(QObject):
    """QComboBox 가 polish 될 때 자동으로 보정을 적용하는 글로벌 필터."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: D401
        if event.type() == QEvent.Type.Polish and isinstance(watched, QComboBox):
            apply_combo_close_fix(watched)
        return False


_GLOBAL_FILTER: Optional[_ComboCloseFilter] = None


def install_combo_close_filter(app: Optional[QApplication] = None) -> None:
    """``QApplication`` 에 글로벌 필터를 한 번만 설치한다."""
    global _GLOBAL_FILTER
    if _GLOBAL_FILTER is not None:
        return
    target = app or QApplication.instance()
    if target is None:
        return
    _GLOBAL_FILTER = _ComboCloseFilter(target)
    target.installEventFilter(_GLOBAL_FILTER)
