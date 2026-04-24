"""Round 107 — ROM 에너지 분석."""

from __future__ import annotations

import numpy as np
import pytest


class TestRomEnergy:
    def test_retention_monotonic(self) -> None:
        from naviertwin.core.analysis.rom_energy import energy_retention

        sv = np.array([10.0, 5.0, 2.0, 0.5, 0.1])
        r = energy_retention(sv)
        assert np.all(np.diff(r) >= 0)
        assert abs(r[-1] - 1.0) < 1e-12

    def test_n_modes(self) -> None:
        from naviertwin.core.analysis.rom_energy import n_modes_for_energy

        sv = np.array([10.0, 5.0, 2.0, 0.5, 0.1])
        n99 = n_modes_for_energy(sv, threshold=0.99)
        assert 1 <= n99 <= len(sv)
        n100 = n_modes_for_energy(sv, threshold=1.0)
        assert n100 == len(sv)

    def test_scree(self) -> None:
        from naviertwin.core.analysis.rom_energy import scree_elbow

        # 큰 점프 있는 구간
        sv = np.array([10.0, 9.5, 9.0, 1.0, 0.5, 0.1])
        e = scree_elbow(sv)
        assert 1 <= e <= len(sv)

    def test_spectrum(self) -> None:
        from naviertwin.core.analysis.rom_energy import energy_spectrum

        sv = np.arange(20, 0, -1, dtype=float)
        s = energy_spectrum(sv)
        assert s["total"] == 1.0
        assert 0 < s["top1"] < s["top3"] < s["top10"] <= 1.0

    def test_empty_raises(self) -> None:
        from naviertwin.core.analysis.rom_energy import energy_retention

        with pytest.raises(ValueError):
            energy_retention(np.array([]))
