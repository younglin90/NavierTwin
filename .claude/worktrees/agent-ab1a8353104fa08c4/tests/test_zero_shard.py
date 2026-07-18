"""Round 413 — ZeRO state shard."""

from __future__ import annotations

import numpy as np


class TestZeroShard:
    def test_round_trip(self) -> None:
        from naviertwin.utils.zero_shard import gather_state, shard_state

        state = {"w": np.arange(20.0), "b": np.arange(8.0)}
        shards = shard_state(state, n_ranks=4)
        out = gather_state(shards)
        assert np.allclose(out["w"], state["w"])
        assert np.allclose(out["b"], state["b"])

    def test_shapes(self) -> None:
        from naviertwin.utils.zero_shard import shard_state

        state = {"w": np.arange(10.0)}
        shards = shard_state(state, n_ranks=3)
        assert sum(len(s["w"]) for s in shards) == 10
