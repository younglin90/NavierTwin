"""v4.1.0 KAN + Tucker 테스트."""

from __future__ import annotations

import numpy as np
import pytest


class TestTucker:
    def test_reconstruction_improves_with_full_ranks(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.tucker_decomp import (
            TuckerDecomposition,
        )

        rng = np.random.default_rng(0)
        T = rng.standard_normal((10, 8, 8))
        # full-rank Tucker = 원본 복원
        tk = TuckerDecomposition(ranks=(10, 8, 8))
        tk.fit(T)
        err = float(np.linalg.norm(T - tk.reconstruct()) / np.linalg.norm(T))
        assert err < 1e-8

    def test_low_rank_compression(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.tucker_decomp import (
            TuckerDecomposition,
        )

        rng = np.random.default_rng(0)
        # 저랭크 합성 텐서
        r = 3
        U = [rng.standard_normal((s, r)) for s in (12, 10, 8)]
        core = rng.standard_normal((r, r, r))
        T = np.einsum("abc,ia,jb,kc->ijk", core, *U)

        tk = TuckerDecomposition(ranks=(r, r, r), max_iter=20)
        tk.fit(T)
        err = float(np.linalg.norm(T - tk.reconstruct()) / np.linalg.norm(T))
        assert err < 1e-6

    def test_rank_dim_mismatch(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.tucker_decomp import (
            TuckerDecomposition,
        )

        tk = TuckerDecomposition(ranks=(3, 3))
        with pytest.raises(ValueError):
            tk.fit(np.zeros((4, 4, 4)))


class TestKANO:
    def test_kano_package_root_export(self) -> None:
        from naviertwin.core.operator_learning.kan import KANO1D
        from naviertwin.core.operator_learning.kan.kano import KANO1D as KANO1DSource

        assert KANO1D is KANO1DSource

    def test_kano_fit_predict(self) -> None:
        pytest.importorskip("torch")
        from naviertwin.core.operator_learning.kan import KANO1D

        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 32, 1)).astype(np.float32)
        Y = np.sin(X).astype(np.float32)
        op = KANO1D(
            in_channels=1, out_channels=1, modes=4, width=8,
            grid_size=5, n_layers=2, max_epochs=2,
        )
        op.fit({"inputs": X, "outputs": Y})
        y_hat = op.predict({"x": X[:2]})
        assert y_hat.shape == (2, 32, 1)
        assert np.all(np.isfinite(y_hat))
