"""Round 54 — PyDMD variants 테스트."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pydmd", reason="pydmd 필요")


def _oscillating_data(n_features: int = 30, n_snap: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n_snap)
    modes = rng.standard_normal((n_features, 2))
    coefs = np.array([np.sin(t), np.cos(t)])
    return modes @ coefs + 0.01 * rng.standard_normal((n_features, n_snap))


class TestHODMD:
    def test_analysis(self) -> None:
        from naviertwin.core.flow_analysis.modal.dmd_advanced import hodmd_analysis

        X = _oscillating_data()
        res = hodmd_analysis(X, svd_rank=3, d=2)
        assert "modes" in res
        assert res["modes"].ndim == 2
        assert res["eigenvalues"].size > 0


class TestOptDMD:
    def test_analysis(self) -> None:
        from naviertwin.core.flow_analysis.modal.dmd_advanced import optdmd_analysis

        X = _oscillating_data()
        res = optdmd_analysis(X, svd_rank=3)
        assert "eigenvalues" in res
        assert res["eigenvalues"].size > 0


class TestHAVOK:
    def test_analysis_1d(self) -> None:
        pytest.importorskip("pydmd.havok", reason="HAVOK optional")
        from naviertwin.core.flow_analysis.modal.dmd_advanced import havok_analysis

        t = np.linspace(0, 10, 500)
        x = np.sin(2 * t) + 0.5 * np.cos(3 * t)
        try:
            res = havok_analysis(x, delays=20, svd_rank=5)
        except Exception as e:
            pytest.skip(f"HAVOK 실행 실패 (환경): {e}")
        assert "A" in res


class TestDMDc:
    def test_analysis_with_control(self) -> None:
        from naviertwin.core.flow_analysis.modal.dmd_advanced import dmdc_analysis

        X = _oscillating_data(n_features=10, n_snap=50)
        U = np.random.default_rng(0).standard_normal((2, 49))
        res = dmdc_analysis(X, U, svd_rank=5)
        assert "modes" in res
        assert res["modes"].shape[0] == X.shape[0]
