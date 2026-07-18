"""Round 315 — xarray/netCDF wrapper."""

from __future__ import annotations

import pytest


class TestXarray:
    def test_has_xarray(self) -> None:
        from naviertwin.core.cfd_reader.xarray_wrapper import has_xarray

        assert isinstance(has_xarray(), bool)

    def test_round_trip(self, tmp_path) -> None:
        pytest.importorskip("xarray")
        pytest.importorskip("netCDF4")
        import numpy as np

        from naviertwin.core.cfd_reader.xarray_wrapper import (
            read_netcdf,
            write_netcdf,
        )

        path = tmp_path / "x.nc"
        write_netcdf(path, {"u": np.arange(5.0), "v": np.zeros(5)})
        out = read_netcdf(path)
        assert "u" in out
        assert np.allclose(out["u"], np.arange(5.0))
