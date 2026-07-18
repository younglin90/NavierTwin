"""Round 318 — SU2 restart writer."""

from __future__ import annotations

import numpy as np


class TestSU2:
    def test_round_trip(self, tmp_path) -> None:
        from naviertwin.core.cfd_reader.su2_writer import (
            read_restart,
            write_restart,
        )

        p = tmp_path / "restart.dat"
        cols = ["x", "y", "rho"]
        data = np.array([[1.0, 2.0, 0.5], [3.0, 4.0, 0.8]])
        write_restart(p, cols, data)
        c2, d2 = read_restart(p)
        assert c2 == cols
        assert np.allclose(d2, data)

    def test_dim_mismatch_raises(self, tmp_path) -> None:
        import pytest

        from naviertwin.core.cfd_reader.su2_writer import write_restart

        with pytest.raises(ValueError):
            write_restart(tmp_path / "x.dat", ["a"], np.zeros((3, 2)))
