"""Round 442 — hash chain audit log."""

from __future__ import annotations


class TestHashChain:
    def test_verify_intact(self) -> None:
        from naviertwin.utils.hash_chain import HashChain

        c = HashChain()
        c.append("e1")
        c.append("e2")
        c.append("e3")
        assert c.verify()

    def test_tampering_detected(self) -> None:
        from naviertwin.utils.hash_chain import HashChain

        c = HashChain()
        c.append("ok")
        c.append("tamper-me")
        # mutate middle entry
        c.entries[0] = ("changed", c.entries[0][1])
        assert not c.verify()
