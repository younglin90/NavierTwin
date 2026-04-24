"""Round 317 — OpenFOAM writer."""

from __future__ import annotations

import numpy as np


class TestFoamWriter:
    def test_scalar(self, tmp_path) -> None:
        from naviertwin.core.cfd_reader.foam_writer import write_internal_scalar

        p = tmp_path / "p"
        write_internal_scalar(p, "p", np.array([1.0, 2.0, 3.0]))
        text = p.read_text()
        assert "internalField" in text
        assert "volScalarField" in text
        assert "1.0" in text or "1.0\n" in text

    def test_vector(self, tmp_path) -> None:
        from naviertwin.core.cfd_reader.foam_writer import write_internal_vector

        p = tmp_path / "U"
        v = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        write_internal_vector(p, "U", v)
        text = p.read_text()
        assert "(1.0 2.0 3.0)" in text
        assert "volVectorField" in text
