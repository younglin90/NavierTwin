"""Round 193 — Reynolds decomposition."""

from __future__ import annotations

import numpy as np


class TestReynolds:
    def test_decompose(self) -> None:
        from naviertwin.core.analysis.reynolds import reynolds_decompose

        rng = np.random.default_rng(0)
        U = 3.0 + rng.standard_normal((200, 5))
        m, f = reynolds_decompose(U)
        assert np.allclose(m, U.mean(axis=0))
        assert np.all(np.abs(f.mean(axis=0)) < 1e-10)

    def test_stress(self) -> None:
        from naviertwin.core.analysis.reynolds import (
            reynolds_decompose,
            reynolds_stress,
        )

        rng = np.random.default_rng(0)
        u = rng.standard_normal((1000, 3))
        v = u * 0.5 + 0.1 * rng.standard_normal((1000, 3))  # 양의 상관
        _, uf = reynolds_decompose(u)
        _, vf = reynolds_decompose(v)
        R = reynolds_stress(uf, vf)
        assert np.all(R > 0)

    def test_tke(self) -> None:
        from naviertwin.core.analysis.reynolds import tke_pointwise

        rng = np.random.default_rng(0)
        U = rng.standard_normal((500, 4))
        V = rng.standard_normal((500, 4))
        k = tke_pointwise(U, V)
        # ≈ ½(1+1) = 1
        assert np.all(np.abs(k - 1.0) < 0.2)

    def test_intensity(self) -> None:
        from naviertwin.core.analysis.reynolds import turbulence_intensity

        rng = np.random.default_rng(0)
        U = 10.0 + rng.standard_normal((1000, 3))
        ti = turbulence_intensity(U)
        assert np.all(ti > 0)
        assert np.all(ti < 0.2)
