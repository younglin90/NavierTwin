"""Round 312 — zarr reader."""

from __future__ import annotations

import pytest


class TestZarr:
    def test_has_zarr(self) -> None:
        from naviertwin.core.cfd_reader.zarr_reader import has_zarr

        assert isinstance(has_zarr(), bool)

    def test_round_trip(self, tmp_path) -> None:
        pytest.importorskip("zarr")
        import numpy as np

        from naviertwin.core.cfd_reader.zarr_reader import read_zarr, write_zarr

        path = tmp_path / "x.zarr"
        write_zarr(path, {"u": np.arange(10.0), "v": np.ones((3, 4))})
        out = read_zarr(path)
        assert np.allclose(out["u"], np.arange(10.0))
        assert out["v"].shape == (3, 4)
