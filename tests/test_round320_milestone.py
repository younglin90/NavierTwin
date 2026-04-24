"""Round 320 — F category milestone: data/IO + PVD writer."""

from __future__ import annotations


class TestMilestoneF:
    def test_imports(self) -> None:
        from naviertwin.core.cfd_reader import (  # noqa: F401
            cgns_advanced,
            fluent_cas_ext,
            foam_writer,
            parquet_reader,
            su2_writer,
            vtk_pvd_writer,
            xarray_wrapper,
            zarr_reader,
        )
        from naviertwin.core.io import async_loader, hdf5_virtual  # noqa: F401

    def test_pvd_e2e(self, tmp_path) -> None:
        from naviertwin.core.cfd_reader.vtk_pvd_writer import write_pvd

        p = tmp_path / "series.pvd"
        write_pvd(p, [(0.0, "step0.vtu"), (1.0, "step1.vtu"), (2.0, "step2.vtu")])
        text = p.read_text()
        assert 'timestep="0.0"' in text
        assert 'file="step1.vtu"' in text
        assert text.count("<DataSet") == 3

    def test_async_e2e(self) -> None:
        from naviertwin.core.io.async_loader import AsyncLoader

        loader = AsyncLoader(iter(range(20)), max_buffer=4)
        assert sum(loader.iter()) == sum(range(20))
