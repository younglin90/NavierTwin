"""경량 이벤트 버스 — 토픽 기반 pub/sub.

GUI 의 Qt 시그널/슬롯과 별개로 core 내부에서도 쓸 수 있는 decoupled 이벤트.

Examples:
    >>> from naviertwin.utils.event_bus import EventBus
    >>> bus = EventBus()
    >>> seen = []
    >>> _ = bus.subscribe("train_step", lambda x: seen.append(x))
    >>> bus.publish("train_step", {"epoch": 1})
    >>> seen
    [{'epoch': 1}]
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Callable
from uuid import uuid4

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

Handler = Callable[[Any], None]


class EventBus:
    """topic → handler 리스트 저장/호출."""

    def __init__(self) -> None:
        self._subs: dict[str, dict[str, Handler]] = {}
        self._lock = Lock()

    def subscribe(self, topic: str, handler: Handler) -> str:
        """handler ID 반환."""
        hid = uuid4().hex
        with self._lock:
            self._subs.setdefault(topic, {})[hid] = handler
        return hid

    def unsubscribe(self, topic: str, handler_id: str) -> bool:
        with self._lock:
            if topic in self._subs and handler_id in self._subs[topic]:
                del self._subs[topic][handler_id]
                return True
        return False

    def publish(self, topic: str, payload: Any = None) -> int:
        """실행된 handler 수."""
        with self._lock:
            handlers = list(self._subs.get(topic, {}).values())
        count = 0
        idx = 0
        while idx < len(handlers):
            h = handlers[idx]
            try:
                h(payload)
                count += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("handler 실패 (topic=%s): %s", topic, e)
            idx += 1
        return count

    def clear(self, topic: str | None = None) -> None:
        with self._lock:
            if topic is None:
                self._subs.clear()
            else:
                self._subs.pop(topic, None)

    def subscriber_count(self, topic: str) -> int:
        with self._lock:
            return len(self._subs.get(topic, {}))


__all__ = ["EventBus"]
