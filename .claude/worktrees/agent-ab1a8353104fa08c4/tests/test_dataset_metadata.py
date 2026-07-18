"""Round 512 — dataset metadata."""

from __future__ import annotations

import pytest


class TestMeta:
    def test_round_trip(self, tmp_path) -> None:
        from naviertwin.utils.dataset_metadata import read_metadata, write_metadata

        m = {"name": "ds1", "version": "1.0", "n_samples": 100, "extra": "x"}
        p = tmp_path / "m.json"
        write_metadata(p, m)
        assert read_metadata(p) == m

    def test_invalid(self, tmp_path) -> None:
        from naviertwin.utils.dataset_metadata import write_metadata

        with pytest.raises(ValueError):
            write_metadata(tmp_path / "bad.json", {"name": "x"})  # missing fields
