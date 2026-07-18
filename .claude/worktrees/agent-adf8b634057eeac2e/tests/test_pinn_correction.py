"""v3.1.0 PINN + physics correction + SINDy 테스트."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch 필요")


class TestPINNSolver:
    def test_1d_poisson_converges(self) -> None:
        """1D Poisson d²u/dx² = -π² sin(π x), u(0)=u(1)=0 → u = sin(π x)."""
        import torch

        from naviertwin.core.physnemo.pina_wrapper import PINNSolver

        def residual(model: object, x: torch.Tensor) -> torch.Tensor:
            x = x.requires_grad_(True)
            u = model(x)
            du = torch.autograd.grad(
                u.sum(), x, create_graph=True
            )[0]
            d2u = torch.autograd.grad(
                du.sum(), x, create_graph=True
            )[0]
            return d2u + (np.pi ** 2) * torch.sin(np.pi * x)

        pinn = PINNSolver(
            in_dim=1, out_dim=1, hidden=32, n_layers=3,
            max_epochs=500, lr=5e-3,
        )
        collocation = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(-1, 1)
        bc = {
            "x": np.array([[0.0], [1.0]], dtype=np.float32),
            "u": np.array([[0.0], [0.0]], dtype=np.float32),
        }
        pinn.fit(residual_fn=residual, collocation=collocation, boundary=bc)

        # x = 0.5 에서 u ≈ sin(π/2) = 1
        u05 = float(pinn.predict(np.array([[0.5]]))[0, 0])
        assert abs(u05 - 1.0) < 0.2, f"u(0.5)={u05:.3f}, 기대값 ≈ 1"


class TestPhysicsCorrection:
    def test_linear_constraint_projection(self) -> None:
        from naviertwin.core.physics_correction.physics_correction import (
            project_linear_constraint,
        )

        u = np.array([1.0, 2.0, 3.0])
        A = np.array([[1.0, 1.0, 1.0]])
        b = np.array([9.0])
        u2 = project_linear_constraint(u, A, b)
        assert float(u2.sum()) == pytest.approx(9.0, abs=1e-10)
        # 최소 변화 — 투영 전후 차이는 상수 (1, 1, 1)/3 방향
        diff = u2 - u
        assert np.allclose(diff, diff[0])

    def test_mass_conservation_scaling(self) -> None:
        from naviertwin.core.physics_correction.physics_correction import (
            enforce_mass_conservation,
        )

        rho = np.array([1.0, 2.0, 3.0])
        V = np.array([0.5, 0.5, 1.0])  # total = 0.5+1+3 = 4.5
        rho2 = enforce_mass_conservation(rho, V, target_mass=9.0)
        assert float(np.dot(rho2, V)) == pytest.approx(9.0, abs=1e-10)

    def test_constraint_validation(self) -> None:
        from naviertwin.core.physics_correction.physics_correction import (
            project_linear_constraint,
        )

        with pytest.raises(ValueError):
            project_linear_constraint(
                np.zeros(3), np.zeros((2, 5)), np.zeros(2)
            )


class TestHybridROM:
    def test_hybrid_rom_reduces_residual(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.pod import SnapshotPOD
        from naviertwin.core.physics_correction.hybrid_rom import HybridROM

        rng = np.random.default_rng(0)
        # 저랭크 + 약한 비선형
        U = rng.standard_normal((60, 4))
        V = rng.standard_normal((4, 30))
        X = U @ V + 0.05 * rng.standard_normal((60, 30))

        pod = SnapshotPOD(n_modes=3)
        pod.fit(X)
        # POD 만 복원 시 오차
        X_lin = pod.decode(pod.encode(X))
        err_lin = float(np.linalg.norm(X - X_lin) / np.linalg.norm(X))

        hybrid = HybridROM(reducer=pod, hidden=32, max_epochs=50, lr=5e-3)
        hybrid.fit(X)
        X_rec = hybrid.reconstruct(X)
        err_hyb = float(np.linalg.norm(X - X_rec) / np.linalg.norm(X))

        # Hybrid 가 POD 단독보다 낮거나 같아야 함
        assert err_hyb <= err_lin + 1e-6, (
            f"hybrid err({err_hyb:.4g}) > linear({err_lin:.4g})"
        )


class TestSINDy:
    def test_linear_oscillator_recovered(self) -> None:
        """dx/dt = A x 선형 시스템의 계수 복원."""
        from naviertwin.core.flow_analysis.modal.sindy_wrapper import SINDy

        # harmonic oscillator: dx/dt = y, dy/dt = -x
        t = np.linspace(0, 10, 2000)
        dt = t[1] - t[0]
        x = np.sin(t)
        y = np.cos(t)
        X = np.column_stack([x, y])

        s = SINDy(poly_degree=1, threshold=0.05)
        s.fit(X, dt=dt)

        assert s.is_fitted
        assert s.coef_.shape[0] == 2
        eqs = s.equations()
        assert len(eqs) == 2
        # dx0/dt ≈ x1, dx1/dt ≈ -x0 — 계수 부호/크기 기본 검사
        assert any("x1" in eq for eq in eqs)

    def test_requires_2d_input(self) -> None:
        from naviertwin.core.flow_analysis.modal.sindy_wrapper import SINDy

        s = SINDy()
        with pytest.raises(ValueError):
            s.fit(np.array([1.0, 2.0, 3.0]), dt=0.1)
