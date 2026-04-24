"""Round 20 — Lie canonicalization + NFNO-DeepONet + HybridROMAdv."""

from __future__ import annotations

import numpy as np
import pytest


class TestSO2Canonicalizer:
    def test_round_trip(self) -> None:
        from naviertwin.core.equivariant.physics_embedded.lie_algebra_no import (
            SO2Canonicalizer,
        )

        rng = np.random.default_rng(0)
        u = rng.standard_normal((10, 2))
        can = SO2Canonicalizer()
        u_c, theta = can.canonicalize(u)
        u_back = can.decanonicalize(u_c, theta)
        assert np.allclose(u, u_back, atol=1e-10)

    def test_equivariant_wrapper_so2(self) -> None:
        """base_fn 이 identity 이면 equivariant wrapper 도 identity 이어야."""
        from naviertwin.core.equivariant.physics_embedded.lie_algebra_no import (
            SO2EquivariantOperator,
        )

        op = SO2EquivariantOperator(base_fn=lambda u: u)
        rng = np.random.default_rng(0)
        u = rng.standard_normal((8, 2))
        out = op(u)
        assert np.allclose(out, u, atol=1e-10)


class TestNFNODeepONet:
    def test_fit_predict(self) -> None:
        pytest.importorskip("torch")
        from naviertwin.core.operator_learning.deeponet.nfno_deeponet import (
            NFNODeepONet,
        )

        rng = np.random.default_rng(0)
        Bx = rng.uniform(-1, 1, (10, 20, 2)).astype(np.float32)
        Bu = rng.standard_normal((10, 20)).astype(np.float32)
        T = rng.uniform(-1, 1, (8, 2)).astype(np.float32)
        Y = rng.standard_normal((10, 8)).astype(np.float32)

        op = NFNODeepONet(
            branch_coord_dim=2, branch_value_dim=1, trunk_in=2,
            hidden=16, latent=8, max_epochs=2,
        )
        op.fit({
            "branch_coords": Bx, "branch_values": Bu,
            "trunk_inputs": T, "outputs": Y,
        })
        out = op.predict({
            "branch_coords": Bx[:2], "branch_values": Bu[:2],
            "trunk_inputs": T,
        })
        assert out.shape == (2, 8)


class TestHybridROMAdv:
    def test_constraint_satisfied(self) -> None:
        pytest.importorskip("torch")
        from naviertwin.core.dimensionality_reduction.linear.pod import SnapshotPOD
        from naviertwin.core.physics_correction.hybrid_rom_adv import HybridROMAdv

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 20))
        # 열마다 합이 0 이도록 보정
        X = X - X.mean(axis=0, keepdims=True)

        pod = SnapshotPOD(n_modes=3)
        pod.fit(X)
        C = np.ones((1, 30))
        d = np.zeros(1)

        m = HybridROMAdv(reducer=pod, C=C, d=d, hidden=16, max_epochs=10, lr=5e-3)
        m.fit(X)
        X_rec = m.reconstruct(X)
        residual = float(np.abs(C @ X_rec - d[:, None]).max())
        assert residual < 1e-8

    def test_C_d_pairing(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.pod import SnapshotPOD
        from naviertwin.core.physics_correction.hybrid_rom_adv import HybridROMAdv

        pod = SnapshotPOD(n_modes=2)
        pod.fit(np.random.default_rng(0).standard_normal((10, 5)))
        with pytest.raises(ValueError, match="C 와 d"):
            HybridROMAdv(reducer=pod, C=np.ones((1, 10)), d=None)
