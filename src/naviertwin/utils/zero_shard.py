"""ZeRO-style optimizer state shard (toy) — partition param tensors among ranks.

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.zero_shard import shard_state
    >>> state = {'p1': np.ones(10), 'p2': np.zeros(20)}
    >>> shards = shard_state(state, n_ranks=2)
    >>> len(shards)
    2
"""

from __future__ import annotations

from typing import Any

import numpy as np


def shard_state(
    state: dict[str, Any], *, n_ranks: int = 2,
) -> list[dict[str, Any]]:
    """각 param tensor 를 n_ranks 등분 (마지막 rank 가 잔여)."""
    shards: list[dict[str, Any]] = []
    rank_idx = 0
    while rank_idx < n_ranks:
        shards.append({})
        rank_idx += 1
    items = list(state.items())
    item_idx = 0
    while item_idx < len(items):
        k, v = items[item_idx]
        v = np.asarray(v)
        chunks = np.array_split(v, n_ranks, axis=0)
        r = 0
        while r < len(chunks):
            c = chunks[r]
            shards[r][k] = c
            r += 1
        item_idx += 1
    return shards


def gather_state(shards: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    keys = list(shards[0].keys())
    out = {}
    key_idx = 0
    while key_idx < len(keys):
        k = keys[key_idx]
        chunks = []
        shard_idx = 0
        while shard_idx < len(shards):
            chunks.append(shards[shard_idx][k])
            shard_idx += 1
        out[k] = np.concatenate(chunks, axis=0)
        key_idx += 1
    return out


__all__ = ["gather_state", "shard_state"]
