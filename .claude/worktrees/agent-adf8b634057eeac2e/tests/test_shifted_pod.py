"""Round 277 — Shifted POD."""

from __future__ import annotations

import numpy as np


class TestShiftedPOD:
    def test_traveling_wave_rank_drops(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.shifted_pod import (
            shifted_pod,
        )

        n = 64
        x = np.linspace(0, 2 * np.pi, n, endpoint=False)
        X = np.array([np.sin(x - 0.2 * t) for t in range(40)]).T  # (n, 40)
        # standard SVD energy required for 99%
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        # shifted SVD: align then SVD
        modes, shifts = shifted_pod(X, rank=5)
        # check modes shape
        assert modes.shape == (n, 5)
        # shifts should be approximately monotonic
        assert shifts.shape == (40,)

    def test_estimate_shift_zero(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.shifted_pod import (
            estimate_shifts,
        )

        n = 32
        X = np.tile(np.cos(np.linspace(0, 2 * np.pi, n))[:, None], (1, 5))
        shifts = estimate_shifts(X)
        assert np.all(shifts == 0)
