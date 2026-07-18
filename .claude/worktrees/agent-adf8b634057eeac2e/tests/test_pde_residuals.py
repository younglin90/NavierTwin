"""Round 115 — PINN PDE 잔차."""

from __future__ import annotations

import pytest


class TestPDEResiduals:
    def test_continuity_incompressible(self) -> None:
        """u=y, v=-x → ∂u/∂x + ∂v/∂y = 0 + 0 = 0."""
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.pde_residuals import continuity_residual_2d

        x = torch.randn(10, 1, requires_grad=True)
        y = torch.randn(10, 1, requires_grad=True)
        u = y
        v = -x
        r = continuity_residual_2d(u, v, x, y)
        assert torch.max(torch.abs(r)).item() < 1e-6

    def test_poisson_analytic(self) -> None:
        """φ = x² + y² → ∇²φ = 4."""
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.pde_residuals import poisson_residual_2d

        x = torch.randn(5, 1, requires_grad=True)
        y = torch.randn(5, 1, requires_grad=True)
        phi = x ** 2 + y ** 2
        source = torch.full_like(phi, 4.0)
        r = poisson_residual_2d(phi, x, y, source)
        assert torch.max(torch.abs(r)).item() < 1e-5

    def test_ns_shape(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.pde_residuals import ns_residual_2d

        x = torch.randn(8, 1, requires_grad=True)
        y = torch.randn(8, 1, requires_grad=True)
        t = torch.randn(8, 1, requires_grad=True)
        u = torch.sin(x) * torch.cos(y) * torch.exp(-t)
        v = -torch.cos(x) * torch.sin(y) * torch.exp(-t)
        p = -0.25 * (torch.cos(2 * x) + torch.cos(2 * y)) * torch.exp(-2 * t)
        Ru, Rv, Rc = ns_residual_2d(u, v, p, x, y, t, rho=1.0, nu=0.01)
        assert Ru.shape == u.shape
        assert Rc.shape == u.shape

    def test_advection_diffusion_constant(self) -> None:
        """c = const → r = 0."""
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.pde_residuals import (
            advection_diffusion_residual_2d,
        )

        x = torch.randn(5, 1, requires_grad=True)
        y = torch.randn(5, 1, requires_grad=True)
        t = torch.randn(5, 1, requires_grad=True)
        c = 3.0 * torch.ones_like(x) + 0 * x + 0 * y + 0 * t
        u = torch.ones_like(x) + 0 * x
        v = torch.zeros_like(x) + 0 * x
        r = advection_diffusion_residual_2d(c, x, y, t, u, v, D=0.01)
        assert torch.max(torch.abs(r)).item() < 1e-5
