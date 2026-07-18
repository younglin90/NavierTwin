"""Round 19 — TNO + FlowDMD + PyKoopman wrapper."""

from __future__ import annotations

import numpy as np
import pytest


class TestTNO:
    def test_shapes(self) -> None:
        pytest.importorskip("torch")
        from naviertwin.core.time_series.temporal_no.tno import TNO

        rng = np.random.default_rng(0)
        seqs = rng.standard_normal((3, 10, 16, 1)).astype(np.float32)
        m = TNO(
            spatial_size=16, channels=1, width=8, modes=4,
            history=3, horizon=2, max_epochs=2,
        )
        m.fit(seqs)
        y = m.predict(seqs[0, :3])
        assert y.shape == (2, 16, 1)

    def test_insufficient_T(self) -> None:
        pytest.importorskip("torch")
        from naviertwin.core.time_series.temporal_no.tno import TNO

        m = TNO(
            spatial_size=8, channels=1, width=4, modes=2,
            history=5, horizon=3, max_epochs=1,
        )
        with pytest.raises(ValueError, match="history"):
            m.fit(np.zeros((2, 6, 8, 1), dtype=np.float32))


class TestFlowDMD:
    def test_rollout_finite(self) -> None:
        pytest.importorskip("torch")
        from naviertwin.core.operator_learning.koopman.flowdmd import FlowDMD

        rng = np.random.default_rng(0)
        seqs = rng.standard_normal((3, 25, 4)).astype(np.float32)
        m = FlowDMD(
            n_features=4, n_blocks=2, hidden=16, dmd_rank=3, max_epochs=2,
        )
        m.fit({"sequences": seqs})
        out = m.predict(seqs[0, 0], n_steps=4)
        assert out.shape == (4, 4)
        assert np.all(np.isfinite(out))
        eigs = m.eigenvalues()
        assert eigs.shape[0] > 0


class TestKoopmanAnalysis:
    def test_recovers_linear_system(self) -> None:
        from naviertwin.core.flow_analysis.modal.pykoopman_wrapper import (
            KoopmanAnalysis,
        )

        A_true = np.array([[0.9, 0.05], [-0.02, 0.85]])
        rng = np.random.default_rng(0)
        X = [rng.standard_normal(2)]
        for _ in range(100):
            X.append(A_true @ X[-1] + 1e-4 * rng.standard_normal(2))
        X = np.array(X)

        ka = KoopmanAnalysis()
        ka.fit(X)
        K = ka.K_
        assert K.shape == (2, 2)
        err = float(np.linalg.norm(K - A_true) / np.linalg.norm(A_true))
        assert err < 0.2

    def test_predict(self) -> None:
        from naviertwin.core.flow_analysis.modal.pykoopman_wrapper import (
            KoopmanAnalysis,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 3))
        ka = KoopmanAnalysis()
        ka.fit(X)
        out = ka.predict(X[-1], n_steps=5)
        assert out.shape == (5, 3)
