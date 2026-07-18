"""경량 Neural ODE — fixed-step RK4 backprop.

주의: adjoint method 없이 backprop-through-time (메모리 O(T)).

Examples:
    >>> import torch  # doctest: +SKIP
"""

from __future__ import annotations


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc


def NeuralODE(state_dim: int, hidden: int = 32):
    _torch()
    import torch
    import torch.nn as nn

    class _ODE(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, state_dim),
            )

        def rhs(self, x):
            return self.f(x)

        def rk4_step(self, x, dt):
            k1 = self.rhs(x)
            k2 = self.rhs(x + 0.5 * dt * k1)
            k3 = self.rhs(x + 0.5 * dt * k2)
            k4 = self.rhs(x + dt * k3)
            return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        def forward(self, x0, t0: float, t1: float, dt: float):
            n = int(round((t1 - t0) / dt))
            x = x0
            traj = [x]
            step = 0
            while step < n:
                x = self.rk4_step(x, dt)
                traj.append(x)
                step += 1
            return torch.stack(traj, dim=0)

    return _ODE()


__all__ = ["NeuralODE"]
