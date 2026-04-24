"""Round 314 — HDF5 virtual dataset."""

from __future__ import annotations

import pytest


class TestHDF5Virtual:
    def test_has_h5py(self) -> None:
        from naviertwin.core.io.hdf5_virtual import has_h5py

        assert isinstance(has_h5py(), bool)

    def test_virtual_round_trip(self, tmp_path) -> None:
        h5py = pytest.importorskip("h5py")
        import numpy as np

        from naviertwin.core.io.hdf5_virtual import create_virtual, read_virtual

        # create 3 source files
        sources = []
        for k in range(3):
            fp = tmp_path / f"src_{k}.h5"
            with h5py.File(fp, "w") as f:
                f.create_dataset("x", data=np.full(5, float(k), dtype="f4"))
            sources.append((str(fp), "x"))
        out = tmp_path / "virt.h5"
        create_virtual(out, sources, shape=(3, 5), dtype="f4")
        arr = read_virtual(out)
        assert arr.shape == (3, 5)
        assert np.allclose(arr[0], 0.0)
        assert np.allclose(arr[2], 2.0)
