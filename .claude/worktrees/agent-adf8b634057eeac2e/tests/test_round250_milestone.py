"""Round 250 — MEGA milestone R241-R249."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

R241_249 = [
    "naviertwin.core.analysis.stft",
    "naviertwin.gui.mode_gallery",
    "naviertwin.utils.config_diff",
    "naviertwin.utils.hashing",
    "naviertwin.utils.disk_cache",
    "naviertwin.utils.watchdog",
    "naviertwin.utils.safe_yaml",
    "naviertwin.utils.safe_toml",
    "naviertwin.core.dimensionality_reduction.linear.incremental_svd",
]


class TestRound250:
    @pytest.mark.parametrize("m", R241_249)
    def test_importable(self, m: str) -> None:
        import importlib
        importlib.import_module(m)

    def test_disk_cache_with_hash(self, tmp_path: Path) -> None:
        from naviertwin.utils.disk_cache import disk_cache
        from naviertwin.utils.hashing import hash_array

        @disk_cache(tmp_path)
        def heavy(x):
            return hash_array(np.arange(x))

        h1 = heavy(100)
        h2 = heavy(100)
        assert h1 == h2

    def test_stft_structured_log(self, tmp_path: Path) -> None:
        from naviertwin.core.analysis.stft import stft
        from naviertwin.utils.structured_log import StructuredLogger

        x = np.sin(2 * np.pi * 10 * np.arange(0, 1, 0.001))
        f, T, Z = stft(x, fs=1000, window=64, overlap=32)
        log = StructuredLogger(tmp_path / "stft.jsonl")
        log.emit("stft_done", n_frames=int(Z.shape[1]), n_freqs=int(Z.shape[0]))
        assert log.read_all()[-1]["n_frames"] == Z.shape[1]
