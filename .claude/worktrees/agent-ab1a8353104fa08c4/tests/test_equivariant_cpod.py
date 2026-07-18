"""v4.2.0 Equivariant FNO + ConstrainedPOD 테스트."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch 필요")


class TestC4EquivariantFNO:
    def test_augmentation_and_prediction(self) -> None:
        from naviertwin.core.equivariant.group_equiv_fno.group_equiv_fno import (
            C4EquivariantFNO2D,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 16, 16, 1)).astype(np.float32)
        Y = X ** 2
        op = C4EquivariantFNO2D(
            in_channels=1, out_channels=1, modes1=4, modes2=4,
            width=8, n_layers=2, max_epochs=2,
        )
        op.fit({"inputs": X, "outputs": Y})
        y_hat = op.predict({"x": X[:2]})
        assert y_hat.shape == (2, 16, 16, 1)

    def test_approx_rotation_equivariance(self) -> None:
        """rotate(predict(x)) 과 predict(rotate(x)) 의 평균 차이가 작아야 함."""
        from naviertwin.core.equivariant.group_equiv_fno.group_equiv_fno import (
            C4EquivariantFNO2D,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((6, 16, 16, 1)).astype(np.float32)
        Y = X  # identity 에 가까운 단순 매핑
        op = C4EquivariantFNO2D(
            in_channels=1, out_channels=1, modes1=4, modes2=4,
            width=8, n_layers=1, max_epochs=2,
        )
        op.fit({"inputs": X, "outputs": Y})

        x_sample = X[:1]
        y1 = op.predict({"x": x_sample})
        y1_rot = np.rot90(y1, k=1, axes=(1, 2))

        x_rot = np.rot90(x_sample, k=1, axes=(1, 2))
        y2 = op.predict({"x": x_rot})

        # 4개 회전 평균 덕분에 두 결과가 가까워야 함
        diff = float(np.linalg.norm(y1_rot - y2))
        ref = float(np.linalg.norm(y2) + 1e-10)
        # 완전 equivariance 가 아닌 근사이므로 느슨한 허용
        assert diff / ref < 1.0


class TestConstrainedPOD:
    def test_sum_zero_constraint_satisfied(self) -> None:
        """제약: Σ_i x_i = 0 → 복원값도 합이 0 이어야."""
        from naviertwin.core.dimensionality_reduction.linear.cpod import ConstrainedPOD

        rng = np.random.default_rng(0)
        n_feat, n_snap = 30, 20
        X = rng.standard_normal((n_feat, n_snap))
        # 원본도 합이 0 이 되도록 중심화
        X = X - X.mean(axis=0, keepdims=True)

        C = np.ones((1, n_feat))
        d = np.zeros(1)

        cpod = ConstrainedPOD(n_modes=5, C=C, d=d)
        cpod.fit(X)
        X_rec = cpod.reconstruct(X)

        residual = np.abs(C @ X_rec - d[:, None]).max()
        assert residual < 1e-8, f"제약 위반: {residual}"

    def test_no_constraint_equivalent_to_pod(self) -> None:
        """C=None 이면 일반 POD 와 같아야."""
        from naviertwin.core.dimensionality_reduction.linear.cpod import ConstrainedPOD
        from naviertwin.core.dimensionality_reduction.linear.pod import SnapshotPOD

        rng = np.random.default_rng(0)
        X = rng.standard_normal((25, 15))
        cpod = ConstrainedPOD(n_modes=4, C=None, d=None)
        cpod.fit(X)
        pod = SnapshotPOD(n_modes=4)
        pod.fit(X)

        # 재구성 오차가 거의 같아야 함 (근사, SVD 수치오차 허용)
        err_cpod = float(np.linalg.norm(X - cpod.reconstruct(X)))
        err_pod = float(np.linalg.norm(X - pod.reconstruct(X)))
        assert abs(err_cpod - err_pod) / max(err_pod, 1e-10) < 0.1

    def test_dimension_validation(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.cpod import ConstrainedPOD

        with pytest.raises(ValueError, match="C"):
            ConstrainedPOD(n_modes=2, C=None, d=np.zeros(1))
