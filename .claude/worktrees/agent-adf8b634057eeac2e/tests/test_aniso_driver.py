"""Round 377 — anisotropic driver."""

from __future__ import annotations

import numpy as np


class TestAnisoDriver:
    def test_target_count(self) -> None:
        from naviertwin.core.amr.aniso_driver import normalize_metric

        M = np.array([np.eye(2), np.eye(2)])
        M2 = normalize_metric(M, n_target=10)
        assert np.isclose(np.sqrt(np.linalg.det(M2)).sum(), 10.0)

    def test_zero_handling(self) -> None:
        from naviertwin.core.amr.aniso_driver import normalize_metric

        M = np.zeros((3, 2, 2))
        M2 = normalize_metric(M, n_target=10)
        assert np.allclose(M2, 0)
