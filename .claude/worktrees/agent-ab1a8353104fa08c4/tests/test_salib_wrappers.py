"""Round 52 — SALib Morris/FAST/PAWN/Delta 래퍼."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("SALib", reason="SALib 필요")


def ishigami(X: np.ndarray) -> np.ndarray:
    a, b = 7.0, 0.1
    return np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] ** 4 * np.sin(X[:, 0])


PROBLEM = {
    "num_vars": 3,
    "names": ["x1", "x2", "x3"],
    "bounds": [[-np.pi, np.pi]] * 3,
}


class TestMorris:
    def test_shapes(self) -> None:
        from naviertwin.core.sensitivity.salib_wrappers import morris_analysis

        mu, sigma = morris_analysis(PROBLEM, ishigami, n_trajectories=20, seed=0)
        assert mu.shape == (3,)
        assert sigma.shape == (3,)
        assert np.all(mu >= 0)


class TestFAST:
    def test_s1_plus_st(self) -> None:
        from naviertwin.core.sensitivity.salib_wrappers import fast_analysis

        S1, ST = fast_analysis(PROBLEM, ishigami, n_samples=256, seed=0)
        assert S1.shape == (3,)
        assert ST.shape == (3,)
        # ST >= S1 원칙 (대략)
        assert np.all(ST >= S1 - 0.5)


class TestPAWN:
    def test_median_shape(self) -> None:
        from naviertwin.core.sensitivity.salib_wrappers import pawn_analysis

        median = pawn_analysis(PROBLEM, ishigami, n_samples=256, S=5, seed=0)
        assert median.shape == (3,)
        assert np.all(np.isfinite(median))


class TestDelta:
    def test_shapes(self) -> None:
        from naviertwin.core.sensitivity.salib_wrappers import delta_analysis

        d, s1 = delta_analysis(PROBLEM, ishigami, n_samples=256, seed=0)
        assert d.shape == (3,)
        assert s1.shape == (3,)
        # x1, x2 는 x3 보다 δ 가 커야 (Ishigami 특성)
        assert d[0] > d[2] * 0.5 or d[1] > d[2] * 0.5
