"""Round 587 — error-path coverage for xarray_wrapper / zarr_reader."""

from __future__ import annotations

import builtins

import pytest


class TestXarrayWrapperErrors:
    def test_has_xarray_returns_bool(self) -> None:
        from naviertwin.core.cfd_reader.xarray_wrapper import has_xarray

        assert isinstance(has_xarray(), bool)

    def test_read_netcdf_no_xarray(self, monkeypatch, tmp_path) -> None:
        from naviertwin.core.cfd_reader import xarray_wrapper

        real_import = builtins.__import__

        def block(name, *a, **kw):
            if name == "xarray":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block)
        with pytest.raises(ImportError, match="xarray"):
            xarray_wrapper.read_netcdf(tmp_path / "x.nc")

    def test_write_netcdf_no_xarray(self, monkeypatch, tmp_path) -> None:
        from naviertwin.core.cfd_reader import xarray_wrapper

        real_import = builtins.__import__

        def block(name, *a, **kw):
            if name == "xarray":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block)
        with pytest.raises(ImportError, match="xarray"):
            xarray_wrapper.write_netcdf(tmp_path / "x.nc", {"a": [1.0, 2.0]})


class TestZarrReaderErrors:
    def test_has_zarr_returns_bool(self) -> None:
        from naviertwin.core.cfd_reader.zarr_reader import has_zarr

        assert isinstance(has_zarr(), bool)

    def test_read_zarr_no_zarr(self, monkeypatch, tmp_path) -> None:
        from naviertwin.core.cfd_reader import zarr_reader

        real_import = builtins.__import__

        def block(name, *a, **kw):
            if name == "zarr":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block)
        with pytest.raises(ImportError, match="zarr"):
            zarr_reader.read_zarr(tmp_path / "x.zarr")

    def test_write_zarr_no_zarr(self, monkeypatch, tmp_path) -> None:
        from naviertwin.core.cfd_reader import zarr_reader

        real_import = builtins.__import__

        def block(name, *a, **kw):
            if name == "zarr":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block)
        with pytest.raises(ImportError, match="zarr"):
            zarr_reader.write_zarr(tmp_path / "x.zarr", {"a": [1, 2, 3]})
