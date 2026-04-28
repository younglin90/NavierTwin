"""Round 617 — phase-locked averaging + cycle extraction + period detection."""

from __future__ import annotations

import numpy as np
import pytest


class TestPhaseAverage:
    def test_recovers_sinusoid(self) -> None:
        from naviertwin.core.flow_analysis.phase_lock import phase_average

        rng = np.random.default_rng(0)
        period = 1.0
        t = np.linspace(0, 100, 5000)
        u = np.sin(2 * np.pi * t / period) + 0.1 * rng.standard_normal(5000)
        phases, mean, rms = phase_average(t, u, period=period, n_bins=36)
        assert phases.shape == (36,)
        assert mean.shape == (36,)
        # 평균이 사인 파형 따라감
        ideal = np.sin(phases)
        # 강한 상관
        cor = np.corrcoef(mean, ideal)[0, 1]
        assert cor > 0.9

    def test_rms_for_pure_signal(self) -> None:
        from naviertwin.core.flow_analysis.phase_lock import phase_average

        # 노이즈 없는 순수 사인: 같은 위상의 모든 샘플 동일 → RMS ≈ 0 (within bin width)
        t = np.linspace(0, 1000, 100000)
        u = np.sin(2 * np.pi * t)
        phases, mean, rms = phase_average(t, u, period=1.0, n_bins=20)
        assert np.all(rms < 0.15)  # 빈 폭만큼 잔여 변동 허용

    def test_2d_signal(self) -> None:
        from naviertwin.core.flow_analysis.phase_lock import phase_average

        rng = np.random.default_rng(1)
        t = np.linspace(0, 50, 2000)
        u = np.outer(np.sin(2 * np.pi * t), np.array([1.0, 2.0, 3.0]))
        u = u + 0.1 * rng.standard_normal((2000, 3))
        phases, mean, rms = phase_average(t, u, period=1.0, n_bins=20)
        assert mean.shape == (20, 3)

    def test_invalid_period_raises(self) -> None:
        from naviertwin.core.flow_analysis.phase_lock import phase_average

        with pytest.raises(ValueError, match="period"):
            phase_average(np.zeros(10), np.zeros(10), period=0.0)

    def test_invalid_n_bins_raises(self) -> None:
        from naviertwin.core.flow_analysis.phase_lock import phase_average

        with pytest.raises(ValueError, match="n_bins"):
            phase_average(np.zeros(10), np.zeros(10), period=1.0, n_bins=0)

    def test_length_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.phase_lock import phase_average

        with pytest.raises(ValueError, match="length"):
            phase_average(np.zeros(10), np.zeros(15), period=1.0)


class TestCycleExtract:
    def test_basic_shape(self) -> None:
        from naviertwin.core.flow_analysis.phase_lock import cycle_extract

        t = np.linspace(0, 5.0, 500)
        u = np.sin(2 * np.pi * t)
        phase_grid, cycles = cycle_extract(t, u, period=1.0, n_phase=20)
        assert phase_grid.shape == (20,)
        # 5개의 완전한 사이클 추출 가능
        assert cycles.shape[0] == 5
        assert cycles.shape[1] == 20

    def test_n_cycles_limit(self) -> None:
        from naviertwin.core.flow_analysis.phase_lock import cycle_extract

        t = np.linspace(0, 10.0, 1000)
        u = np.sin(2 * np.pi * t)
        phase_grid, cycles = cycle_extract(
            t, u, period=1.0, n_phase=10, n_cycles=3,
        )
        assert cycles.shape[0] == 3

    def test_too_short_raises(self) -> None:
        from naviertwin.core.flow_analysis.phase_lock import cycle_extract

        t = np.linspace(0, 0.5, 100)  # 주기 1.0보다 짧음
        u = np.sin(2 * np.pi * t)
        with pytest.raises(ValueError, match="too short"):
            cycle_extract(t, u, period=1.0)

    def test_invalid_period_raises(self) -> None:
        from naviertwin.core.flow_analysis.phase_lock import cycle_extract

        t = np.linspace(0, 1, 100)
        with pytest.raises(ValueError, match="period"):
            cycle_extract(t, np.zeros(100), period=0.0)

    def test_n_phase_too_small_raises(self) -> None:
        from naviertwin.core.flow_analysis.phase_lock import cycle_extract

        t = np.linspace(0, 5, 500)
        with pytest.raises(ValueError, match="n_phase"):
            cycle_extract(t, np.zeros(500), period=1.0, n_phase=1)

    def test_2d_signal(self) -> None:
        from naviertwin.core.flow_analysis.phase_lock import cycle_extract

        t = np.linspace(0, 5.0, 500)
        u = np.outer(np.sin(2 * np.pi * t), np.array([1.0, 2.0]))
        phase_grid, cycles = cycle_extract(t, u, period=1.0, n_phase=10)
        assert cycles.shape == (5, 10, 2)


class TestFundamentalPeriod:
    def test_known_period_sine(self) -> None:
        from naviertwin.core.flow_analysis.phase_lock import (
            fundamental_period_from_acf,
        )

        period = 0.5
        t = np.linspace(0, 10, 2000)
        u = np.sin(2 * np.pi * t / period)
        T = fundamental_period_from_acf(t, u)
        assert abs(T - period) / period < 0.05

    def test_too_short_raises(self) -> None:
        from naviertwin.core.flow_analysis.phase_lock import (
            fundamental_period_from_acf,
        )

        with pytest.raises(ValueError, match="too short"):
            fundamental_period_from_acf(np.zeros(5), np.zeros(5))

    def test_constant_signal_raises(self) -> None:
        from naviertwin.core.flow_analysis.phase_lock import (
            fundamental_period_from_acf,
        )

        t = np.linspace(0, 10, 100)
        with pytest.raises(ValueError, match="constant"):
            fundamental_period_from_acf(t, np.ones(100))

    def test_non_uniform_t_raises(self) -> None:
        from naviertwin.core.flow_analysis.phase_lock import (
            fundamental_period_from_acf,
        )

        # 동일 시간 반복 (median diff = 0)
        t = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        u = np.sin(t)
        with pytest.raises(ValueError, match="non-uniform"):
            fundamental_period_from_acf(t, u)
