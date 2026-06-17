"""Euler-Maruyama SDE — dx = f_θ(x, t) dt + g dW.

drift f_θ 는 MLP, diffusion g 는 scalar 또는 MLP.

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


def NeuralSDE(
    state_dim: int, hidden: int = 32, diffusion: float = 0.1,
):
    _torch()
    import torch
    import torch.nn as nn

    class _SDE(nn.Module):
        def __init__(self):
            super().__init__()
            self.drift = nn.Sequential(
                nn.Linear(state_dim + 1, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, state_dim),
            )
            self.g = float(diffusion)

        def step(self, x, t, dt):
            """Euler-Maruyama 1 step."""
            tb = torch.full((x.size(0), 1), float(t), device=x.device, dtype=x.dtype)
            f = self.drift(torch.cat([x, tb], dim=-1))
            noise = torch.randn_like(x) * (self.g * (dt ** 0.5))
            return x + f * dt + noise

        def simulate(self, x0, t0: float, t1: float, dt: float):
            n = int((t1 - t0) / dt)
            x = x0
            traj = [x]
            k = 0
            while k < n:
                t = t0 + k * dt
                x = self.step(x, t, dt)
                traj.append(x)
                k += 1
            return torch.stack(traj, dim=0)

        def forward(self, x):
            return self.drift(x)

    return _SDE()


__all__ = ["NeuralSDE"]
