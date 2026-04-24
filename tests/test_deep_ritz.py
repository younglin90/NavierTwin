"""Round 48 — Deep Ritz solver."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch 필요")


class TestDeepRitz:
    def test_poisson_1d_converges(self) -> None:
        import torch

        from naviertwin.core.physnemo.deep_ritz import DeepRitzSolver

        def f(x: torch.Tensor) -> torch.Tensor:
            return (np.pi ** 2) * torch.sin(np.pi * x)

        solver = DeepRitzSolver(
            in_dim=1, hidden=32, n_layers=3,
            n_collocation=64, max_epochs=400, lr=5e-3, w_bc=20.0,
        )
        solver.fit(
            f, domain_range=(0.0, 1.0),
            boundary_x=np.array([[0.0], [1.0]], dtype=np.float32),
            boundary_u=np.array([[0.0], [0.0]], dtype=np.float32),
        )
        # u(0.5) ≈ sin(π/2) = 1
        u05 = float(solver.predict(np.array([[0.5]], dtype=np.float32))[0, 0])
        assert abs(u05 - 1.0) < 0.5  # Deep Ritz 수렴은 PINN 보다 느릴 수 있음

    def test_unfitted_raises(self) -> None:
        from naviertwin.core.physnemo.deep_ritz import DeepRitzSolver

        solver = DeepRitzSolver()
        with pytest.raises(RuntimeError):
            solver.predict(np.zeros((1, 1)))
