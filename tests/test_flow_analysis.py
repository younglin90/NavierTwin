"""유동 분석 테스트 모음.

numpy 만으로 실행 가능한 테스트를 포함한다.
scipy 가 있으면 PSD 관련 추가 검증을 수행한다.
pyvista 가 있으면 Q-criterion/λ₂ 메쉬 테스트도 수행한다.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _make_sine_signal(
    freq: float = 10.0, dt: float = 0.001, duration: float = 2.0
) -> tuple[NDArray[np.float64], float]:
    """단일 주파수 정현파 신호를 생성한다.

    Returns:
        (signal, dt) 튜플.
    """
    t = np.arange(0, duration, dt)
    signal = np.sin(2 * np.pi * freq * t).astype(np.float64)
    return signal, dt


# ---------------------------------------------------------------------------
# FFT 테스트
# ---------------------------------------------------------------------------


def test_fft_basic() -> None:
    """단일 정현파의 FFT 에서 올바른 주파수 피크가 나타나는지 검증한다."""
    from naviertwin.core.flow_analysis.statistics.fft_psd import compute_fft

    freq_true = 10.0
    dt = 0.001
    signal, _ = _make_sine_signal(freq=freq_true, dt=dt, duration=2.0)

    freqs, amps = compute_fft(signal, dt)

    # 양수 주파수만 반환
    assert len(freqs) > 0
    assert np.all(freqs >= 0.0)

    # 최대 진폭 위치가 10 Hz 근처
    peak_freq = freqs[np.argmax(amps)]
    assert abs(peak_freq - freq_true) < 1.0  # 1 Hz 이내


def test_fft_two_frequencies() -> None:
    """두 주파수 성분이 동시에 나타나는지 검증한다."""
    from naviertwin.core.flow_analysis.statistics.fft_psd import compute_fft

    dt = 0.001
    duration = 2.0
    t = np.arange(0, duration, dt)
    f1, f2 = 5.0, 20.0
    signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

    freqs, amps = compute_fft(signal, dt)

    # f1, f2 근처에서 높은 진폭
    idx1 = np.argmin(np.abs(freqs - f1))
    idx2 = np.argmin(np.abs(freqs - f2))
    assert amps[idx1] > 0.3
    assert amps[idx2] > 0.1


def test_fft_invalid_dt() -> None:
    """dt <= 0 이면 ValueError 가 발생해야 한다."""
    from naviertwin.core.flow_analysis.statistics.fft_psd import compute_fft

    signal = np.ones(100)
    with pytest.raises(ValueError, match="dt"):
        compute_fft(signal, dt=0.0)

    with pytest.raises(ValueError, match="dt"):
        compute_fft(signal, dt=-0.01)


def test_fft_empty_signal() -> None:
    """빈 신호에 ValueError 가 발생해야 한다."""
    from naviertwin.core.flow_analysis.statistics.fft_psd import compute_fft

    with pytest.raises(ValueError):
        compute_fft(np.array([]), dt=0.01)


def test_fft_dc_component() -> None:
    """DC 성분(평균값)이 f=0 에 나타나는지 검증한다."""
    from naviertwin.core.flow_analysis.statistics.fft_psd import compute_fft

    signal = np.ones(1000) * 3.0  # DC only
    freqs, amps = compute_fft(signal, dt=0.001)

    assert freqs[0] == pytest.approx(0.0)
    assert amps[0] == pytest.approx(3.0, rel=1e-3)


# ---------------------------------------------------------------------------
# PSD 테스트
# ---------------------------------------------------------------------------


def test_psd_basic() -> None:
    """정현파 신호의 PSD 가 해당 주파수에서 피크를 보이는지 검증한다."""
    from naviertwin.core.flow_analysis.statistics.fft_psd import compute_psd

    freq_true = 15.0
    dt = 0.001
    signal, _ = _make_sine_signal(freq=freq_true, dt=dt, duration=5.0)

    freqs, psd = compute_psd(signal, dt, window="hann")

    assert len(freqs) == len(psd)
    assert np.all(psd >= 0.0)

    peak_freq = freqs[np.argmax(psd)]
    assert abs(peak_freq - freq_true) < 2.0


def test_psd_invalid_dt() -> None:
    """dt <= 0 이면 ValueError 가 발생해야 한다."""
    from naviertwin.core.flow_analysis.statistics.fft_psd import compute_psd

    with pytest.raises(ValueError):
        compute_psd(np.ones(100), dt=0.0)


def test_psd_nonnegative() -> None:
    """PSD 값은 항상 0 이상이어야 한다."""
    from naviertwin.core.flow_analysis.statistics.fft_psd import compute_psd

    rng = np.random.default_rng(42)
    signal = rng.standard_normal(512)
    _, psd = compute_psd(signal, dt=0.01)

    assert np.all(psd >= 0.0)


# ---------------------------------------------------------------------------
# find_dominant_frequencies 테스트
# ---------------------------------------------------------------------------


def test_find_dominant_frequencies() -> None:
    """지배 주파수 피크가 올바르게 탐색되는지 검증한다."""
    from naviertwin.core.flow_analysis.statistics.fft_psd import (
        compute_fft,
        find_dominant_frequencies,
    )

    dt = 0.001
    duration = 2.0
    t = np.arange(0, duration, dt)
    f1, f2, f3 = 5.0, 15.0, 30.0
    signal = (
        1.0 * np.sin(2 * np.pi * f1 * t)
        + 0.5 * np.sin(2 * np.pi * f2 * t)
        + 0.2 * np.sin(2 * np.pi * f3 * t)
    )

    freqs, amps = compute_fft(signal, dt)
    peaks = find_dominant_frequencies(freqs, amps, n_peaks=3)

    assert len(peaks) <= 3
    assert len(peaks) > 0

    # 각 피크는 frequency, amplitude, strouhal 키를 가짐
    for peak in peaks:
        assert "frequency" in peak
        assert "amplitude" in peak
        assert "strouhal" in peak
        assert peak["frequency"] > 0.0
        assert peak["amplitude"] > 0.0

    # 첫 번째 피크가 가장 강한 주파수(f1=5 Hz)와 가까워야 함
    top_freq = peaks[0]["frequency"]
    assert abs(top_freq - f1) < 2.0


def test_find_dominant_frequencies_n_peaks() -> None:
    """n_peaks 개수를 초과하여 반환하지 않는지 검증한다."""
    from naviertwin.core.flow_analysis.statistics.fft_psd import (
        compute_fft,
        find_dominant_frequencies,
    )

    dt = 0.01
    t = np.arange(0, 5, dt)
    signal = np.sin(2 * np.pi * 1 * t)
    freqs, amps = compute_fft(signal, dt)

    peaks = find_dominant_frequencies(freqs, amps, n_peaks=2)
    assert len(peaks) <= 2


def test_find_dominant_frequencies_empty() -> None:
    """주파수 배열이 비어있으면 빈 리스트를 반환해야 한다."""
    from naviertwin.core.flow_analysis.statistics.fft_psd import (
        find_dominant_frequencies,
    )

    result = find_dominant_frequencies(
        np.array([0.0]),  # DC 만 존재
        np.array([1.0]),
        n_peaks=5,
    )
    assert result == []


# ---------------------------------------------------------------------------
# compute_field_fft 테스트
# ---------------------------------------------------------------------------


def test_field_fft_spatial_mean() -> None:
    """공간 평균 FFT 가 올바르게 동작하는지 검증한다."""
    from naviertwin.core.flow_analysis.statistics.fft_psd import (
        compute_field_fft,
    )

    n_time, n_points = 200, 50
    dt = 0.01
    # 모든 점에서 동일한 주파수
    t = np.arange(n_time) * dt
    snapshots = np.outer(np.sin(2 * np.pi * 5 * t), np.ones(n_points))

    freqs, amps = compute_field_fft(snapshots, dt, point_idx=None)

    assert len(freqs) == len(amps)
    peak_freq = freqs[np.argmax(amps)]
    assert abs(peak_freq - 5.0) < 1.0


def test_field_fft_specific_point() -> None:
    """특정 점 인덱스로 FFT 를 계산할 수 있는지 검증한다."""
    from naviertwin.core.flow_analysis.statistics.fft_psd import (
        compute_field_fft,
    )

    n_time, n_points = 100, 20
    dt = 0.01
    rng = np.random.default_rng(0)
    snapshots = rng.standard_normal((n_time, n_points))

    freqs, amps = compute_field_fft(snapshots, dt, point_idx=5)
    assert len(freqs) > 0


def test_field_fft_out_of_range() -> None:
    """point_idx 가 범위를 벗어나면 IndexError 가 발생해야 한다."""
    from naviertwin.core.flow_analysis.statistics.fft_psd import (
        compute_field_fft,
    )

    snapshots = np.ones((50, 10))
    with pytest.raises(IndexError):
        compute_field_fft(snapshots, dt=0.01, point_idx=100)


def test_field_fft_vector_field() -> None:
    """3D 벡터 필드 (N_time, N_points, 3) 를 처리할 수 있는지 검증한다."""
    from naviertwin.core.flow_analysis.statistics.fft_psd import (
        compute_field_fft,
    )

    snapshots = np.ones((50, 10, 3))
    freqs, amps = compute_field_fft(snapshots, dt=0.01)
    assert len(freqs) > 0


# ---------------------------------------------------------------------------
# y+ 테스트
# ---------------------------------------------------------------------------


def test_yplus_formula() -> None:
    """y+ = u_tau * y / nu 공식이 올바르게 구현되었는지 검증한다."""
    from naviertwin.core.flow_analysis.boundary_layer.yplus import (
        compute_yplus,
    )

    rho = 1.225
    nu = 1.5e-5
    # tau_w = 1.0 Pa → u_tau = sqrt(1.0 / 1.225)
    tau_w = np.array([[1.0, 0.0, 0.0]])
    y_wall = np.array([1e-4])

    u_tau_expected = math.sqrt(1.0 / rho)
    y_plus_expected = u_tau_expected * y_wall[0] / nu

    y_plus = compute_yplus(tau_w, rho=rho, nu=nu, y_wall=y_wall)

    assert y_plus[0] == pytest.approx(y_plus_expected, rel=1e-5)


def test_yplus_invalid_rho() -> None:
    """rho <= 0 이면 ValueError 가 발생해야 한다."""
    from naviertwin.core.flow_analysis.boundary_layer.yplus import (
        compute_yplus,
    )

    tau_w = np.array([[1.0, 0.0, 0.0]])
    y_wall = np.array([1e-4])

    with pytest.raises(ValueError, match="rho"):
        compute_yplus(tau_w, rho=0.0, nu=1.5e-5, y_wall=y_wall)

    with pytest.raises(ValueError, match="rho"):
        compute_yplus(tau_w, rho=-1.0, nu=1.5e-5, y_wall=y_wall)


def test_yplus_invalid_nu() -> None:
    """nu <= 0 이면 ValueError 가 발생해야 한다."""
    from naviertwin.core.flow_analysis.boundary_layer.yplus import (
        compute_yplus,
    )

    tau_w = np.array([[1.0, 0.0, 0.0]])
    y_wall = np.array([1e-4])

    with pytest.raises(ValueError, match="nu"):
        compute_yplus(tau_w, rho=1.225, nu=0.0, y_wall=y_wall)


def test_yplus_multiple_points() -> None:
    """여러 벽면 점에 대해 y+ 를 일괄 계산할 수 있는지 검증한다."""
    from naviertwin.core.flow_analysis.boundary_layer.yplus import (
        compute_yplus,
    )

    n = 10
    tau_w = np.zeros((n, 3))
    tau_w[:, 0] = np.linspace(0.1, 1.0, n)
    y_wall = np.ones(n) * 1e-4

    y_plus = compute_yplus(tau_w, rho=1.225, nu=1.5e-5, y_wall=y_wall)
    assert y_plus.shape == (n,)
    assert np.all(y_plus >= 0.0)


# ---------------------------------------------------------------------------
# estimate_first_cell_height 테스트
# ---------------------------------------------------------------------------


def test_estimate_first_cell_height() -> None:
    """estimate_first_cell_height 가 양수 값을 반환하는지 검증한다."""
    from naviertwin.core.flow_analysis.boundary_layer.yplus import (
        estimate_first_cell_height,
    )

    y1 = estimate_first_cell_height(
        y_plus_target=1.0,
        Re=1e6,
        L=1.0,
        nu=1.5e-5,
        rho=1.225,
        U_inf=15.0,
    )

    assert y1 > 0.0
    # 전형적인 y+=1 첫 번째 셀 높이는 수 마이크로미터 ~ 수십 마이크로미터
    assert 1e-7 < y1 < 1e-2


def test_estimate_first_cell_height_larger_yplus() -> None:
    """y+=30 이 y+=1 보다 큰 셀 높이를 반환하는지 검증한다."""
    from naviertwin.core.flow_analysis.boundary_layer.yplus import (
        estimate_first_cell_height,
    )

    kwargs: dict = dict(Re=1e6, L=1.0, nu=1.5e-5, rho=1.225, U_inf=15.0)

    y1_fine = estimate_first_cell_height(y_plus_target=1.0, **kwargs)
    y1_coarse = estimate_first_cell_height(y_plus_target=30.0, **kwargs)

    assert y1_coarse > y1_fine
    assert y1_coarse == pytest.approx(30.0 * y1_fine, rel=1e-3)


def test_estimate_first_cell_height_invalid_input() -> None:
    """입력이 0 이하이면 ValueError 가 발생해야 한다."""
    from naviertwin.core.flow_analysis.boundary_layer.yplus import (
        estimate_first_cell_height,
    )

    with pytest.raises(ValueError):
        estimate_first_cell_height(
            y_plus_target=1.0,
            Re=0.0,
            L=1.0,
            nu=1.5e-5,
            rho=1.225,
            U_inf=15.0,
        )

    with pytest.raises(ValueError):
        estimate_first_cell_height(
            y_plus_target=1.0,
            Re=1e6,
            L=1.0,
            nu=-1.0,
            rho=1.225,
            U_inf=15.0,
        )


# ---------------------------------------------------------------------------
# compute_friction_velocity 테스트
# ---------------------------------------------------------------------------


def test_compute_friction_velocity() -> None:
    """u_tau = sqrt(|tau_w| / rho) 공식 검증."""
    from naviertwin.core.flow_analysis.boundary_layer.yplus import (
        compute_friction_velocity,
    )

    rho = 1.225
    tau_w = np.array([[3.0, 4.0, 0.0]])  # |tau_w| = 5 Pa
    u_tau = compute_friction_velocity(tau_w, rho)

    expected = math.sqrt(5.0 / rho)
    assert u_tau[0] == pytest.approx(expected, rel=1e-5)


def test_compute_friction_velocity_zero_stress() -> None:
    """전단응력이 0 이면 u_tau = 0 이어야 한다."""
    from naviertwin.core.flow_analysis.boundary_layer.yplus import (
        compute_friction_velocity,
    )

    tau_w = np.zeros((5, 3))
    u_tau = compute_friction_velocity(tau_w, rho=1.0)

    np.testing.assert_array_equal(u_tau, 0.0)


def test_compute_friction_velocity_invalid_rho() -> None:
    """rho <= 0 이면 ValueError 가 발생해야 한다."""
    from naviertwin.core.flow_analysis.boundary_layer.yplus import (
        compute_friction_velocity,
    )

    with pytest.raises(ValueError, match="rho"):
        compute_friction_velocity(np.array([[1.0, 0.0, 0.0]]), rho=-1.0)


# ---------------------------------------------------------------------------
# compute_wall_units 테스트
# ---------------------------------------------------------------------------


def test_compute_wall_units() -> None:
    """compute_wall_units 가 y+ 와 delta_nu 를 올바르게 반환하는지 검증한다."""
    from naviertwin.core.flow_analysis.boundary_layer.yplus import (
        compute_wall_units,
    )

    nu = 1.5e-5
    u_tau = np.array([0.5, 1.0])
    y_wall = np.array([1e-4, 1e-4])

    result = compute_wall_units(y_wall, u_tau, nu)

    assert "y_plus" in result
    assert "delta_nu" in result

    # y+ = u_tau * y / nu
    expected_yp = u_tau * y_wall / nu
    np.testing.assert_allclose(result["y_plus"], expected_yp, rtol=1e-6)

    # delta_nu = nu / u_tau
    expected_dn = nu / u_tau
    np.testing.assert_allclose(result["delta_nu"], expected_dn, rtol=1e-6)


def test_compute_wall_units_zero_utau() -> None:
    """u_tau = 0 인 점에 대해 y+ = 0, delta_nu = 0 이어야 한다."""
    from naviertwin.core.flow_analysis.boundary_layer.yplus import (
        compute_wall_units,
    )

    u_tau = np.array([0.0, 1.0])
    y_wall = np.array([1e-4, 1e-4])

    result = compute_wall_units(y_wall, u_tau, nu=1.5e-5)

    assert result["y_plus"][0] == pytest.approx(0.0)
    assert result["delta_nu"][0] == pytest.approx(0.0)


def test_compute_wall_units_invalid_nu() -> None:
    """nu <= 0 이면 ValueError 가 발생해야 한다."""
    from naviertwin.core.flow_analysis.boundary_layer.yplus import (
        compute_wall_units,
    )

    with pytest.raises(ValueError, match="nu"):
        compute_wall_units(np.array([1e-4]), np.array([0.5]), nu=0.0)


# ---------------------------------------------------------------------------
# Q-criterion / lambda2 (pyvista 필요, 없으면 skip)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    pytest.importorskip("pyvista", reason="pyvista 없음") is None,
    reason="pyvista 없음",
)
def test_q_criterion_no_velocity_field() -> None:
    """속도 필드가 없는 메쉬에 compute_q_criterion 을 호출하면 KeyError."""
    pv = pytest.importorskip("pyvista")
    from naviertwin.core.flow_analysis.vortex.q_criterion import (
        compute_q_criterion,
    )

    mesh = pv.Sphere().cast_to_unstructured_grid()
    # 'U' 필드 없음

    with pytest.raises((KeyError, Exception)):
        compute_q_criterion(mesh, velocity_name="U")


@pytest.mark.skipif(
    pytest.importorskip("pyvista", reason="pyvista 없음") is None,
    reason="pyvista 없음",
)
def test_lambda2_adds_field() -> None:
    """compute_lambda2 가 'lambda2' 필드를 추가하는지 검증한다."""
    pv = pytest.importorskip("pyvista")
    from naviertwin.core.flow_analysis.vortex.q_criterion import (
        compute_lambda2,
    )

    mesh = pv.Sphere().cast_to_unstructured_grid()
    n = mesh.n_points
    rng = np.random.default_rng(0)
    mesh.point_data["U"] = rng.standard_normal((n, 3)).astype(np.float32)

    result = compute_lambda2(mesh, velocity_name="U")

    assert "lambda2" in result.point_data
    assert result.point_data["lambda2"].shape == (n,)
