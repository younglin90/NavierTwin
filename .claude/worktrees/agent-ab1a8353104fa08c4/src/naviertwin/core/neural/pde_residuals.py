"""PINN PDE 잔차 — 2D 비압축 Navier-Stokes + 포아송 + 이류확산.

PyTorch autograd 로 미분 계산 (시공간 입력에서 grad).

Examples:
    >>> import torch  # doctest: +SKIP
    >>> from naviertwin.core.neural.pde_residuals import ns_residual_2d  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc


def _grad(y: Any, x: Any, create_graph: bool = True) -> Any:
    torch = _torch()
    if not y.requires_grad:
        return torch.zeros_like(x)
    g = torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=create_graph, retain_graph=True, allow_unused=True,
    )[0]
    if g is None:
        g = torch.zeros_like(x)
    return g


def continuity_residual_2d(u: Any, v: Any, x: Any, y: Any) -> Any:
    """∂u/∂x + ∂v/∂y."""
    du_dx = _grad(u, x)
    dv_dy = _grad(v, y)
    return du_dx + dv_dy


def ns_residual_2d(
    u: Any, v: Any, p: Any, x: Any, y: Any, t: Any,
    rho: float = 1.0, nu: float = 0.01,
) -> tuple[Any, Any, Any]:
    """2D 비압축 Navier-Stokes 잔차 (Ru, Rv, Rc)."""
    u_t = _grad(u, t)
    u_x = _grad(u, x)
    u_y = _grad(u, y)
    u_xx = _grad(u_x, x)
    u_yy = _grad(u_y, y)

    v_t = _grad(v, t)
    v_x = _grad(v, x)
    v_y = _grad(v, y)
    v_xx = _grad(v_x, x)
    v_yy = _grad(v_y, y)

    p_x = _grad(p, x)
    p_y = _grad(p, y)

    Ru = u_t + u * u_x + v * u_y + (1.0 / rho) * p_x - nu * (u_xx + u_yy)
    Rv = v_t + u * v_x + v * v_y + (1.0 / rho) * p_y - nu * (v_xx + v_yy)
    Rc = u_x + v_y
    return Ru, Rv, Rc


def poisson_residual_2d(phi: Any, x: Any, y: Any, source: Any) -> Any:
    """∇²φ = f → R = φ_xx + φ_yy - f."""
    phi_x = _grad(phi, x)
    phi_xx = _grad(phi_x, x)
    phi_y = _grad(phi, y)
    phi_yy = _grad(phi_y, y)
    return phi_xx + phi_yy - source


def advection_diffusion_residual_2d(
    c: Any, x: Any, y: Any, t: Any, u: Any, v: Any, D: float = 0.01,
) -> Any:
    """c_t + u c_x + v c_y - D(c_xx+c_yy) = 0."""
    c_t = _grad(c, t)
    c_x = _grad(c, x)
    c_xx = _grad(c_x, x)
    c_y = _grad(c, y)
    c_yy = _grad(c_y, y)
    return c_t + u * c_x + v * c_y - D * (c_xx + c_yy)


__all__ = [
    "continuity_residual_2d",
    "ns_residual_2d",
    "poisson_residual_2d",
    "advection_diffusion_residual_2d",
]
