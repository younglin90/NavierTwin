"""CGNS advanced — zone iterator + multi-base traversal (h5py 기반).

CGNS 는 HDF5 위에 spec 으로 zone/family/family_BC 구조를 가진다.

Examples:
    >>> from naviertwin.core.cfd_reader import has_h5py
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
    """Yield (base_name, zone_name, attrs) per zone."""
    if not has_h5py():
        raise ImportError("h5py not installed")
    import h5py
    with h5py.File(str(path), "r") as f:
        base_names = list(f.keys())
        base_idx = 0
        while base_idx < len(base_names):
            base_name = base_names[base_idx]
            base = f[base_name]
            if not isinstance(base, h5py.Group):
                base_idx += 1
                continue
            zone_names = list(base.keys())
            zone_idx = 0
            while zone_idx < len(zone_names):
                zone_name = zone_names[zone_idx]
                zone = base[zone_name]
                if isinstance(zone, h5py.Group):
                    attrs = dict(zone.attrs)
                    yield base_name, zone_name, attrs
                zone_idx += 1
            base_idx += 1


def list_zones(path: str | Path) -> list[tuple[str, str]]:
    zones: list[tuple[str, str]] = []
    zone_iter = iter_zones(path)
    while True:
        try:
            base_name, zone_name, _ = next(zone_iter)
        except StopIteration:
            break
        zones.append((base_name, zone_name))
    return zones


__all__ = ["has_h5py", "iter_zones", "list_zones"]
