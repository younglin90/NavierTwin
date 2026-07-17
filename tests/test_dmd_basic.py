"""Round 584 — coverage uplift for DMDAnalyzer (was 29%)."""

from __future__ import annotations

import numpy as np
import pytest


class TestDMDAnalyzer:
    def test_invalid_method(self) -> None:
        from naviertwin.core.flow_analysis.modal.dmd import DMDAnalyzer

        with pytest.raises(ValueError, match="지원되지 않는"):
            DMDAnalyzer(method="bogus")

    def test_supported_methods_init(self) -> None:
        from naviertwin.core.flow_analysis.modal.dmd import DMDAnalyzer

        for m in ["dmd", "fbdmd", "hodmd", "spdmd"]:
            a = DMDAnalyzer(method=m, dt=0.05, n_modes=3)
            assert a.method == m
            assert a.dt == 0.05
            assert a.n_modes == 3

    def test_fit_or_skip(self) -> None:
        pytest.importorskip("pydmd")
        from naviertwin.core.flow_analysis.modal.dmd import DMDAnalyzer

        # build a low-rank oscillating snapshot matrix
        t = np.linspace(0, 2, 100)
        x = np.linspace(0, 1, 30)
        X = np.outer(np.sin(2 * np.pi * x), np.cos(2 * np.pi * t)) + \
            0.5 * np.outer(np.cos(3 * np.pi * x), np.sin(3 * np.pi * t))
        a = DMDAnalyzer(method="dmd", dt=t[1] - t[0])
        a.fit(X)
        assert a._is_fitted


def _traveling_wave_snapshots(n_time: int = 60):
    """DMD 가 이론적으로 정확히 맞추는 데이터 — 공간모드 2개 × 고유 주파수.

    실수 진행파라 켤레쌍 때문에 실제 랭크는 4 다 (svd_rank 자동이 이를 잡는다).
    """
    x = np.linspace(0, 2 * np.pi, 128)
    t = np.linspace(0, 6, n_time)
    xg, tg = np.meshgrid(x, t, indexing="ij")
    X = np.exp(-0.10 * tg) * np.sin(xg - 1.3 * tg) + np.exp(-0.30 * tg) * np.sin(
        2 * xg - 2.7 * tg
    )
    return X, t


class TestDMDAccuracy:
    """정확도 회귀 — 기존 테스트는 fit 만 하고 재구성 품질을 안 봐서
    기본값 method 가 발산하는 것을 놓쳤다 (fbdmd → dmd 로 수정)."""

    def test_default_method_reconstructs_accurately(self) -> None:
        pytest.importorskip("pydmd")
        from naviertwin.core.flow_analysis.modal.dmd import DMDAnalyzer

        X, t = _traveling_wave_snapshots()
        a = DMDAnalyzer(dt=float(t[1] - t[0]))  # 기본 method
        a.fit(X)
        rel = np.linalg.norm(a.reconstruct(t) - X) / np.linalg.norm(X)
        assert rel < 1e-6, f"기본 method 재구성이 부정확: rel={rel}"

    def test_default_method_recovers_frequencies(self) -> None:
        pytest.importorskip("pydmd")
        from naviertwin.core.flow_analysis.modal.dmd import DMDAnalyzer

        X, t = _traveling_wave_snapshots()
        a = DMDAnalyzer(dt=float(t[1] - t[0]))
        a.fit(X)
        freqs = sorted({round(abs(f), 3) for f in a.frequencies})
        # 참값: 1.3/2pi = 0.207, 2.7/2pi = 0.430
        assert any(abs(f - 0.207) < 0.02 for f in freqs), freqs
        assert any(abs(f - 0.430) < 0.02 for f in freqs), freqs

    def test_reconstruct_extrapolates_beyond_training(self) -> None:
        """DMD 계열의 존재 이유 — 학습에 없던 미래 구간을 맞춘다."""
        pytest.importorskip("pydmd")
        from naviertwin.core.flow_analysis.modal.dmd import DMDAnalyzer

        X, t = _traveling_wave_snapshots()
        half = len(t) // 2
        a = DMDAnalyzer(dt=float(t[1] - t[0]))
        a.fit(X[:, :half])
        future = a.reconstruct(t[half:])
        rel = np.linalg.norm(future - X[:, half:]) / np.linalg.norm(X[:, half:])
        assert rel < 1e-6, f"미래 외삽이 부정확: rel={rel}"
