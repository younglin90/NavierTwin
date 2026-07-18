"""Round 311 — parquet reader."""

from __future__ import annotations

import pytest


class TestParquet:
    def test_has_pyarrow_returns_bool(self) -> None:
        from naviertwin.core.cfd_reader.parquet_reader import has_pyarrow

        assert isinstance(has_pyarrow(), bool)

    def test_round_trip(self, tmp_path) -> None:
        pa = pytest.importorskip("pyarrow")
        np = pytest.importorskip("numpy")
        from naviertwin.core.cfd_reader.parquet_reader import (
            read_parquet,
            write_parquet,
        )

        path = tmp_path / "x.parquet"
        data = {"u": [1.0, 2.0, 3.0], "v": [10, 20, 30]}
        write_parquet(path, data)
        out = read_parquet(path)
        assert "u" in out
        assert np.allclose(out["u"], [1.0, 2.0, 3.0])
        assert pa.__version__  # ensure import worked
