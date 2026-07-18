"""Round 416 — activation offload."""

from __future__ import annotations

import numpy as np


class TestOffload:
    def test_round_trip(self, tmp_path) -> None:
        from naviertwin.utils.activation_offload import OffloadStore

        store = OffloadStore(tmp_path)
        k = store.save(np.arange(10.0))
        x = store.load(k)
        assert np.allclose(x, np.arange(10.0))
        store.free(k)
        assert not (tmp_path / f"act_{k}.npy").exists()
