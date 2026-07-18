"""Activation offload — write tensor to disk during forward, load on backward.

Examples:
    >>> import numpy as np
    >>> import tempfile, pathlib
    >>> from naviertwin.utils.activation_offload import OffloadStore
    >>> with tempfile.TemporaryDirectory() as d:
    ...     store = OffloadStore(pathlib.Path(d))
    ...     k = store.save(np.arange(5))
    ...     out = store.load(k)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class OffloadStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._next = 0

    def save(self, x: NDArray) -> int:
        k = self._next
        self._next += 1
        np.save(self.root / f"act_{k}.npy", np.asarray(x))
        return k

    def load(self, k: int) -> NDArray:
        return np.load(self.root / f"act_{k}.npy")

    def free(self, k: int) -> None:
        p = self.root / f"act_{k}.npy"
        if p.exists():
            p.unlink()


__all__ = ["OffloadStore"]
