"""Round 98 — event bus."""

from __future__ import annotations


class TestEventBus:
    def test_pub_sub(self) -> None:
        from naviertwin.utils.event_bus import EventBus

        bus = EventBus()
        seen: list = []
        hid = bus.subscribe("t", seen.append)
        assert bus.publish("t", 42) == 1
        assert seen == [42]
        assert bus.unsubscribe("t", hid) is True
        assert bus.publish("t", 99) == 0

    def test_multiple(self) -> None:
        from naviertwin.utils.event_bus import EventBus

        bus = EventBus()
        a, b = [], []
        bus.subscribe("x", a.append)
        bus.subscribe("x", b.append)
        bus.publish("x", "msg")
        assert a == ["msg"] and b == ["msg"]
        assert bus.subscriber_count("x") == 2

    def test_handler_exception_isolated(self) -> None:
        from naviertwin.utils.event_bus import EventBus

        bus = EventBus()
        ok: list = []

        def bad(_):
            raise RuntimeError("boom")

        bus.subscribe("t", bad)
        bus.subscribe("t", ok.append)
        n = bus.publish("t", "x")
        assert n == 1  # bad 실패, ok 성공
        assert ok == ["x"]

    def test_clear(self) -> None:
        from naviertwin.utils.event_bus import EventBus

        bus = EventBus()
        bus.subscribe("a", lambda _: None)
        bus.subscribe("b", lambda _: None)
        bus.clear("a")
        assert bus.subscriber_count("a") == 0
        assert bus.subscriber_count("b") == 1
        bus.clear()
        assert bus.subscriber_count("b") == 0
