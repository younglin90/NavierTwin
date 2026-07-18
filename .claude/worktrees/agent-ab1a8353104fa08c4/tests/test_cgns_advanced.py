"""Round 316 — CGNS zone iterator."""

from __future__ import annotations

import pytest


class TestCGNSAdvanced:
    def test_iter_synthetic_zones(self, tmp_path) -> None:
        h5py = pytest.importorskip("h5py")

        from naviertwin.core.cfd_reader import list_zones

        path = tmp_path / "test.cgns"
        with h5py.File(path, "w") as f:
            base = f.create_group("Base#1")
            base.create_group("Zone1")
            base.create_group("Zone2")
            base2 = f.create_group("Base#2")
            base2.create_group("ZoneA")
        zones = list_zones(path)
        assert len(zones) == 3
        # base ordering may vary; check sets
        names = {z for _, z in zones}
        assert names == {"Zone1", "Zone2", "ZoneA"}
