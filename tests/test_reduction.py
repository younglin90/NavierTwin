"""차원 축소 모듈 테스트.

SnapshotPOD 및 RandomizedPOD의 fit/encode/decode 왕복 정확도,
에너지 누적 기여율 단조성, 예외 처리를 검증한다.
모든 테스트는 실제 CFD 데이터 없이 numpy 랜덤 데이터로 동작한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from naviertwin.core.dimensionality_reduction.linear.pod import SnapshotPOD
from naviertwin.core.dimensionality_reduction.linear.randomized_svd import RandomizedPOD


# ---------------------------------------------------------------------------
# 공통 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture()
def snapshot_matrix() -> np.ndarray:
    """재현 가능한 랜덤 스냅샷 행렬 (n_features=200, n_snapshots=50)."""
    rng = np.random.default_rng(42)
    # 저차원 구조를 가진 스냅샷: rank-5 행렬 + 노이즈
    U = rng.standard_normal((200, 5))
    V = rng.standard_normal((5, 50))
    noise = rng.standard_normal((200, 50)) * 0.01
    return (U @ V + noise).astype(np.float64)


# ---------------------------------------------------------------------------
# SnapshotPOD 테스트
# ---------------------------------------------------------------------------


class TestSnapshotPOD:
    """SnapshotPOD 단위 테스트."""

    def test_fit_sets_is_fitted(self, snapshot_matrix: np.ndarray) -> None:
        """fit() 후 is_fitted가 True로 설정되는지 확인한다."""
        pod = SnapshotPOD(n_modes=5)
        assert not pod.is_fitted
        pod.fit(snapshot_matrix)
        assert pod.is_fitted

    def test_fit_shapes(self, snapshot_matrix: np.ndarray) -> None:
        """fit() 후 modes_, singular_values_, mean_의 shape이 올바른지 확인한다."""
        n_modes = 5
        pod = SnapshotPOD(n_modes=n_modes)
        pod.fit(snapshot_matrix)

        n_features = snapshot_matrix.shape[0]
        assert pod.modes_ is not None
        assert pod.modes_.shape == (n_features, n_modes)
        assert pod.singular_values_ is not None
        assert pod.singular_values_.shape == (n_modes,)
        assert pod.mean_ is not None
        assert pod.mean_.shape == (n_features,)

    def test_fit_encode_decode_roundtrip(self, snapshot_matrix: np.ndarray) -> None:
        """fit → encode → decode 왕복 오차가 충분히 작은지 확인한다.

        저차원 구조를 가진 스냅샷이므로 5개 모드로 거의 완벽하게 복원 가능.
        """
        pod = SnapshotPOD(n_modes=5, center=True)
        pod.fit(snapshot_matrix)

        coeffs = pod.encode(snapshot_matrix)   # (n_snapshots, n_modes)
        reconstructed = pod.decode(coeffs)      # (n_features, n_snapshots)

        # 상대 오차 계산
        rel_error = np.linalg.norm(snapshot_matrix - reconstructed) / np.linalg.norm(
            snapshot_matrix
        )
        assert rel_error < 0.1, f"왕복 상대 오차가 너무 큽니다: {rel_error:.6f}"

    def test_encode_output_shape(self, snapshot_matrix: np.ndarray) -> None:
        """encode() 출력의 shape이 (n_snapshots, n_modes)인지 확인한다."""
        n_modes = 7
        pod = SnapshotPOD(n_modes=n_modes)
        pod.fit(snapshot_matrix)

        coeffs = pod.encode(snapshot_matrix)
        n_snapshots = snapshot_matrix.shape[1]
        assert coeffs.shape == (n_snapshots, n_modes), (
            f"encode 출력 shape 불일치: {coeffs.shape} != {(n_snapshots, n_modes)}"
        )

    def test_decode_output_shape(self, snapshot_matrix: np.ndarray) -> None:
        """decode() 출력의 shape이 (n_features, n_snapshots)인지 확인한다."""
        n_modes = 5
        pod = SnapshotPOD(n_modes=n_modes)
        pod.fit(snapshot_matrix)

        coeffs = pod.encode(snapshot_matrix)
        rec = pod.decode(coeffs)
        assert rec.shape == snapshot_matrix.shape, (
            f"decode 출력 shape 불일치: {rec.shape} != {snapshot_matrix.shape}"
        )

    def test_snapshot_pod_energy_ratio(self, snapshot_matrix: np.ndarray) -> None:
        """에너지 누적 기여율이 단조 증가(비감소)하는지 확인한다."""
        pod = SnapshotPOD(n_modes=10)
        pod.fit(snapshot_matrix)

        er = pod.energy_ratio
        assert er.shape == (10,), f"energy_ratio shape 불일치: {er.shape}"

        # 단조 비감소 확인
        diffs = np.diff(er)
        assert np.all(diffs >= -1e-10), (
            f"에너지 기여율이 단조 증가하지 않습니다: {diffs}"
        )

        # 모든 값이 [0, 1] 범위 내
        assert np.all(er >= 0.0), "에너지 기여율에 음수 값이 있습니다."
        assert np.all(er <= 1.0 + 1e-10), "에너지 기여율이 1을 초과합니다."

        # 마지막 값이 첫 번째 값보다 크거나 같음
        assert er[-1] >= er[0], "마지막 누적 에너지가 첫 값보다 작습니다."

    def test_reconstruct_with_fewer_modes(self, snapshot_matrix: np.ndarray) -> None:
        """n_modes를 줄이면 오차가 커지는지 확인한다."""
        pod = SnapshotPOD(n_modes=10)
        pod.fit(snapshot_matrix)

        rec_full = pod.reconstruct(snapshot_matrix, n_modes=10)
        rec_partial = pod.reconstruct(snapshot_matrix, n_modes=2)

        err_full = np.linalg.norm(snapshot_matrix - rec_full)
        err_partial = np.linalg.norm(snapshot_matrix - rec_partial)

        assert err_partial >= err_full, (
            "모드 수를 줄였을 때 오차가 감소했습니다 (예상: 증가)."
        )

    def test_no_center(self, snapshot_matrix: np.ndarray) -> None:
        """center=False 옵션이 정상 동작하는지 확인한다."""
        pod = SnapshotPOD(n_modes=5, center=False)
        pod.fit(snapshot_matrix)
        assert pod.mean_ is not None
        assert np.allclose(pod.mean_, 0.0), "center=False 시 mean_이 0이어야 합니다."

        coeffs = pod.encode(snapshot_matrix)
        rec = pod.decode(coeffs)
        assert rec.shape == snapshot_matrix.shape

    def test_not_fitted_error(self) -> None:
        """fit() 전에 encode() 호출 시 RuntimeError가 발생하는지 확인한다."""
        pod = SnapshotPOD(n_modes=5)
        X = np.random.standard_normal((50, 20))
        with pytest.raises(RuntimeError, match="fit"):
            pod.encode(X)

    def test_pod_insufficient_modes(self) -> None:
        """n_modes > n_snapshots 일 때 경고를 내고 n_modes를 줄이는지 확인한다.

        (ValueError를 내는 대신 자동으로 n_modes를 조정해야 함)
        """
        rng = np.random.default_rng(99)
        X = rng.standard_normal((50, 5))  # n_snapshots = 5
        pod = SnapshotPOD(n_modes=20)     # n_modes > n_snapshots

        # 에러 없이 실행되어야 하며, n_components가 min(n_modes, n_snapshots)으로 설정됨
        pod.fit(X)
        assert pod.n_components <= 5, (
            f"n_components({pod.n_components})가 n_snapshots(5)보다 큽니다."
        )

    def test_invalid_input_ndim(self) -> None:
        """1D 입력에 fit() 호출 시 ValueError가 발생하는지 확인한다."""
        pod = SnapshotPOD(n_modes=3)
        with pytest.raises(ValueError):
            pod.fit(np.ones(50))


# ---------------------------------------------------------------------------
# RandomizedPOD 테스트
# ---------------------------------------------------------------------------


class TestRandomizedPOD:
    """RandomizedPOD 단위 테스트."""

    def test_fit_sets_is_fitted(self, snapshot_matrix: np.ndarray) -> None:
        """fit() 후 is_fitted가 True로 설정되는지 확인한다."""
        rpod = RandomizedPOD(n_modes=5)
        rpod.fit(snapshot_matrix)
        assert rpod.is_fitted

    def test_fit_encode_decode_shapes(self, snapshot_matrix: np.ndarray) -> None:
        """encode/decode 출력 shape이 올바른지 확인한다."""
        n_modes = 5
        rpod = RandomizedPOD(n_modes=n_modes)
        rpod.fit(snapshot_matrix)

        coeffs = rpod.encode(snapshot_matrix)
        n_snapshots = snapshot_matrix.shape[1]
        assert coeffs.shape == (n_snapshots, n_modes)

        rec = rpod.decode(coeffs)
        assert rec.shape == snapshot_matrix.shape

    def test_randomized_pod_consistency(self, snapshot_matrix: np.ndarray) -> None:
        """RandomizedPOD가 SnapshotPOD와 유사한 재구성 오차를 주는지 확인한다.

        랜덤화 SVD는 근사값이므로 SnapshotPOD보다 다소 큰 오차를 허용한다.
        """
        n_modes = 5
        pod = SnapshotPOD(n_modes=n_modes)
        rpod = RandomizedPOD(n_modes=n_modes, random_state=42)

        pod.fit(snapshot_matrix)
        rpod.fit(snapshot_matrix)

        # 각각 재구성 후 오차 비교
        rec_pod = pod.reconstruct(snapshot_matrix)
        rec_rpod = rpod.reconstruct(snapshot_matrix)

        err_pod = np.linalg.norm(snapshot_matrix - rec_pod)
        err_rpod = np.linalg.norm(snapshot_matrix - rec_rpod)

        # RandomizedPOD의 오차가 SnapshotPOD의 5배 이내여야 함
        assert err_rpod <= err_pod * 5.0 + 1e-6, (
            f"RandomizedPOD 오차({err_rpod:.4f})가 SnapshotPOD({err_pod:.4f})보다 "
            f"너무 큽니다."
        )

    def test_energy_ratio_monotone(self, snapshot_matrix: np.ndarray) -> None:
        """RandomizedPOD의 에너지 기여율이 단조 비감소인지 확인한다."""
        rpod = RandomizedPOD(n_modes=8, random_state=0)
        rpod.fit(snapshot_matrix)

        er = rpod.energy_ratio
        diffs = np.diff(er)
        assert np.all(diffs >= -1e-10), (
            f"RandomizedPOD 에너지 기여율이 단조 증가하지 않습니다: {diffs}"
        )

    def test_insufficient_modes(self) -> None:
        """n_modes > n_snapshots 시 n_modes를 자동 조정하는지 확인한다."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((30, 4))
        rpod = RandomizedPOD(n_modes=50)
        rpod.fit(X)
        assert rpod.n_components <= 4
