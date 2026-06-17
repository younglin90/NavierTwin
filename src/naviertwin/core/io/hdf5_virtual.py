"""HDF5 virtual dataset — 여러 파일을 가상 dataset 으로 묶기.

Examples:
    >>> from naviertwin.core.io.hdf5_virtual import has_h5py
    >>> isinstance(has_h5py(), bool)
    True
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def has_h5py() -> bool:
    try:
        import h5py  # noqa: F401
    except ImportError:
        return False
    return True


def create_virtual(
    out_path: str | Path,
    sources: list[tuple[str, str]],
    shape: tuple[int, ...],
    dtype: Any = "f4",
    *,
    name: str = "data",
) -> None:
    """sources = [(file_path, dataset_name), ...] 를 stack 으로 가상 결합."""
    if not has_h5py():
        raise ImportError("h5py not installed; pip install h5py")
    import h5py
    layout = h5py.VirtualLayout(shape=shape, dtype=dtype)
    k = 0
    while k < len(sources):
        fp, dn = sources[k]
        vsource = h5py.VirtualSource(str(fp), dn, shape=shape[1:])
        layout[k] = vsource
        k += 1
    with h5py.File(str(out_path), "w") as f:
        f.create_virtual_dataset(name, layout)


def read_virtual(path: str | Path, name: str = "data") -> Any:
    if not has_h5py():
        raise ImportError("h5py not installed; pip install h5py")
    import h5py
    import numpy as np
    with h5py.File(str(path), "r") as f:
        return np.asarray(f[name][:])


__all__ = ["create_virtual", "has_h5py", "read_virtual"]
