"""Round 536 — memory budget."""

from __future__ import annotations


class TestMemBudget:
    def test_basic(self) -> None:
        from naviertwin.utils.memory_budget import estimate_memory

        r = estimate_memory(n_params=1000, batch=4, seq_len=10, hidden=8,
                              bytes_per_param=4, act_factor=1.0, opt_states=2)
        # params=4000; grads=4000; opt=8000; act=4*10*8*4=1280
        assert r["params"] == 4000
        assert r["grads"] == 4000
        assert r["optimizer"] == 8000
        assert r["activations"] == 1280
        assert r["total"] == 4000 + 4000 + 8000 + 1280
