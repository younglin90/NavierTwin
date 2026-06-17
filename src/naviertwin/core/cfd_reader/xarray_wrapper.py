"""xarray / netCDF wrapper — xarray 옵션, NumPy fallback.

Examples:
    >>> from naviertwin.core.cfd_reader.xarray_wrapper import has_xarray
    >>> isinstance(has_xarray(), bool)
    True
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def has_xarray() -> bool:
    try:
        import xarray  # noqa: F401
    except ImportError:
        return False
    return True


def read_netcdf(path: str | Path) -> dict[str, Any]:
    if not has_xarray():
        raise ImportError("xarray not installed; pip install xarray netcdf4")
    import xarray as xr
    ds = xr.open_dataset(str(path))
    out = dict(map(lambda var: (var, ds[var].values), ds.data_vars))
    ds.close()
    return out


def write_netcdf(path: str | Path, data: dict[str, Any]) -> None:
    if not has_xarray():
        raise ImportError("xarray not installed; pip install xarray netcdf4")
    import xarray as xr
    ds = xr.Dataset(dict(map(lambda item: (item[0], ("dim_0", item[1])), data.items())))
    ds.to_netcdf(str(path))


__all__ = ["has_xarray", "read_netcdf", "write_netcdf"]
