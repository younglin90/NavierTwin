"""CGNS advanced — zone iterator + multi-base traversal (h5py 기반).

CGNS 는 HDF5 위에 spec 으로 zone/family/family_BC 구조를 가진다.

Examples:
    >>> from naviertwin.core.cfd_reader.cgns_advanced import has_h5py
    >>> isinstance(has_h5py(), bool)
    True
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any


def has_h5py() -> bool:
    try:
        import h5py  # noqa: F401
    except ImportError:
        return False
    return True


def iter_zones(path: str | Path) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """yield (base_name, zone_name, attrs) for each zone."""
    if not has_h5py():
        raise ImportError("h5py not installed")
    import h5py
    with h5py.File(str(path), "r") as f:
        for base_name in f:
            base = f[base_name]
            if not isinstance(base, h5py.Group):
                continue
            for zone_name in base:
                zone = base[zone_name]
                if isinstance(zone, h5py.Group):
                    attrs = dict(zone.attrs)
                    yield base_name, zone_name, attrs


def list_zones(path: str | Path) -> list[tuple[str, str]]:
    return [(b, z) for b, z, _ in iter_zones(path)]


__all__ = ["has_h5py", "iter_zones", "list_zones"]
