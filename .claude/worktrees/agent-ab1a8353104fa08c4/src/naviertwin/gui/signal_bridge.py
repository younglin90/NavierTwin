"""GUI 시그널 브리지 — core 의 EventBus 를 Qt Signal 로 중계.

core 모듈은 Qt 의존성 없이 EventBus 에 publish.
GUI 쪽에서 SignalBridge 로 주제별 Qt signal 수신.

Examples:
    >>> # 테스트에서는 Qt 없이 fallback 경로만 사용
    >>> from naviertwin.gui.signal_bridge import SignalBridge
"""

from __future__ import annotations

from typing import Any, Callable

from naviertwin.utils.event_bus import EventBus
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class SignalBridge:
    """EventBus topic → callback 연결. Qt 가 있으면 QSignal 래핑, 없으면 직접 호출."""

    def __init__(self, bus: EventBus | None = None) -> None:
        self.bus = bus or EventBus()
        self._handler_ids: dict[str, list[str]] = {}
        self._qt = self._try_qt()

    @staticmethod
    def _try_qt():
        try:
            from PySide6.QtCore import QObject, Signal

            return (QObject, Signal)
        except ImportError:
            return None

    def connect(self, topic: str, callback: Callable[[Any], None]) -> str:
        """topic 이벤트 → callback 연결."""
        hid = self.bus.subscribe(topic, callback)
        self._handler_ids.setdefault(topic, []).append(hid)
        return hid

    def disconnect(self, topic: str, handler_id: str) -> bool:
        ok = self.bus.unsubscribe(topic, handler_id)
        if ok and topic in self._handler_ids:
            try:
                self._handler_ids[topic].remove(handler_id)
            except ValueError:
                pass
        return ok

    def emit(self, topic: str, payload: Any = None) -> int:
        return self.bus.publish(topic, payload)

    def topics(self) -> list[str]:
        return list(self._handler_ids.keys())

    @property
    def has_qt(self) -> bool:
        return self._qt is not None


__all__ = ["SignalBridge"]
