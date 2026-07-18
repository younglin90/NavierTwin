"""Round 150 — GUI signal bridge."""

from __future__ import annotations


class TestSignalBridge:
    def test_connect_emit(self) -> None:
        from naviertwin.gui.signal_bridge import SignalBridge

        b = SignalBridge()
        seen: list = []
        b.connect("t", seen.append)
        b.emit("t", {"v": 1})
        assert seen == [{"v": 1}]

    def test_disconnect(self) -> None:
        from naviertwin.gui.signal_bridge import SignalBridge

        b = SignalBridge()
        hid = b.connect("topic", lambda _: None)
        assert b.disconnect("topic", hid) is True
        assert b.emit("topic", None) == 0

    def test_multiple_topics(self) -> None:
        from naviertwin.gui.signal_bridge import SignalBridge

        b = SignalBridge()
        a, c = [], []
        b.connect("a", a.append)
        b.connect("c", c.append)
        b.emit("a", 1)
        b.emit("c", 2)
        assert a == [1]
        assert c == [2]
        assert set(b.topics()) == {"a", "c"}
