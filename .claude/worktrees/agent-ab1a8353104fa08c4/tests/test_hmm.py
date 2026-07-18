"""Round 462 — HMM."""

from __future__ import annotations

import numpy as np


class TestHMM:
    def test_linear_micro(self) -> None:
        from naviertwin.core.multiscale.hmm import hmm_macro_flux

        out = hmm_macro_flux(np.array([1.0, 2.0, 3.0]), lambda u: 0.5 * u)
        assert np.allclose(out, [0.5, 1.0, 1.5])

    def test_nonlinear(self) -> None:
        from naviertwin.core.multiscale.hmm import hmm_macro_flux

        out = hmm_macro_flux(np.arange(4.0), lambda u: u * u)
        assert np.allclose(out, [0, 1, 4, 9])
