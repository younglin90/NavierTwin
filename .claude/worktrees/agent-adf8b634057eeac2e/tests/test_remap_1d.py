"""Round 387 — conservative remap."""

from __future__ import annotations

import numpy as np


class TestRemap:
    def test_total_mass_conserved(self) -> None:
        from naviertwin.core.geometry.remap_1d import conservative_remap_1d

        x_old = np.linspace(0, 1, 11)
        u_old = np.arange(10, dtype=float)
        x_new = np.linspace(0, 1, 6)
        u_new = conservative_remap_1d(x_old, u_old, x_new)
        # mass on old: ∫ u dx = sum(u * Δx_old)
        m_old = (u_old * np.diff(x_old)).sum()
        m_new = (u_new * np.diff(x_new)).sum()
        assert np.isclose(m_old, m_new, atol=1e-12)

    def test_uniform_preserved(self) -> None:
        from naviertwin.core.geometry.remap_1d import conservative_remap_1d

        x_old = np.linspace(0, 1, 8)
        u_old = 2.0 * np.ones(7)
        x_new = np.linspace(0, 1, 4)
        u_new = conservative_remap_1d(x_old, u_old, x_new)
        assert np.allclose(u_new, 2.0)
