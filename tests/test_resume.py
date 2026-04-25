"""Round 525 — resume hook."""

from __future__ import annotations

from pathlib import Path


class TestResume:
    def test_latest(self, tmp_path) -> None:
        from naviertwin.utils.workflow.resume import find_latest_ckpt

        for n in [1, 5, 3]:
            (tmp_path / f"ckpt_{n:04d}.bin").write_bytes(b"")
        latest = find_latest_ckpt(tmp_path)
        assert latest is not None
        assert latest.name == "ckpt_0005.bin"

    def test_empty(self, tmp_path) -> None:
        from naviertwin.utils.workflow.resume import find_latest_ckpt

        assert find_latest_ckpt(tmp_path) is None
        _ = Path  # silence unused
