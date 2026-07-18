"""Round 393 — PPM."""

from __future__ import annotations

import numpy as np


class TestPPM:
    def test_linear_face_values(self) -> None:
        from naviertwin.core.solvers.ppm import ppm_face_values

        u = np.array([0., 1, 2, 3, 4])
        uL, uR = ppm_face_values(u)
        # linear: uL ≈ 1.5, uR ≈ 2.5
        assert np.isclose(uL, 1.5)
        assert np.isclose(uR, 2.5)

    def test_monotonize_extremum(self) -> None:
        from naviertwin.core.solvers.ppm import ppm_monotonize

        # local max at u_i, both faces lower
        uL, uR = ppm_monotonize(1.0, 5.0, 1.0, uL=4.0, uR=4.0)
        assert uL == 5.0 and uR == 5.0
