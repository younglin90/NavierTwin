"""Checkpoint manager — best-N retention.

Examples:
    >>> import tempfile, pathlib
    >>> from naviertwin.utils.workflow.checkpoint_mgr import CheckpointManager
    >>> with tempfile.TemporaryDirectory() as d:
    ...     m = CheckpointManager(pathlib.Path(d), keep=2)
    ...     m.add(score=0.5, payload=b'x')
"""

from __future__ import annotations

import json
from pathlib import Path


class CheckpointManager:
    def __init__(self, root: str | Path, *, keep: int = 3, mode: str = "max") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self.mode = mode
        self.index_path = self.root / "index.json"
        self.index: list[dict] = []
        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text())

    def add(self, *, score: float, payload: bytes) -> Path:
        n = len(self.index)
        path = self.root / f"ckpt_{n:04d}.bin"
        path.write_bytes(payload)
        self.index.append({"path": str(path), "score": float(score)})
        self._prune()
        self.index_path.write_text(json.dumps(self.index))
        return path

    def _prune(self) -> None:
        sorted_idx = sorted(self.index, key=lambda r: r["score"],
                              reverse=(self.mode == "max"))
        keep = sorted_idx[:self.keep]
        drop = sorted_idx[self.keep:]
        idx = 0
        while idx < len(drop):
            d = drop[idx]
            p = Path(d["path"])
            if p.exists():
                p.unlink()
            idx += 1
        self.index = keep


__all__ = ["CheckpointManager"]
