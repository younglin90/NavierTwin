"""Round 543 — replay buffer."""

from __future__ import annotations


class TestReplay:
    def test_capacity(self) -> None:
        from naviertwin.core.twin.replay_buffer import ReplayBuffer

        r = ReplayBuffer(capacity=3)
        for k in range(5):
            r.add(k, k * 2)
        assert len(r) == 3
        assert r.latest(1)[0] == (4, 8)

    def test_all(self) -> None:
        from naviertwin.core.twin.replay_buffer import ReplayBuffer

        r = ReplayBuffer(capacity=10)
        r.add(1.0, "a")
        r.add(2.0, "b")
        assert r.all() == [(1.0, "a"), (2.0, "b")]
