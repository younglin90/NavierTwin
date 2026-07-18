"""신경망 빌딩 블록 — MLP / ResidualMLP / SirenMLP.

PINN/DeepONet/surrogate backbone 으로 재사용 가능.

Examples:
    >>> import torch  # doctest: +SKIP
    >>> from naviertwin.core.neural.mlp_blocks import MLP  # doctest: +SKIP
    >>> m = MLP(in_dim=3, out_dim=2, hidden=[32, 32])  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Sequence


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc


def MLP(
    in_dim: int, out_dim: int, hidden: Sequence[int] = (64, 64),
    activation: str = "relu",
):
    """표준 MLP — nn.Sequential."""
    _torch()
    import torch.nn as nn

    acts = {
        "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU,
        "silu": nn.SiLU, "elu": nn.ELU,
    }
    if activation not in acts:
        raise ValueError(f"activation ∈ {list(acts)}")
    layers: list = []
    prev = int(in_dim)
    hidden_idx = 0
    while hidden_idx < len(hidden):
        h = hidden[hidden_idx]
        layers.append(nn.Linear(prev, int(h)))
        layers.append(acts[activation]())
        prev = int(h)
        hidden_idx += 1
    layers.append(nn.Linear(prev, int(out_dim)))
    return nn.Sequential(*layers)


def ResidualMLP(
    in_dim: int, out_dim: int, hidden: int = 64, n_blocks: int = 3,
    activation: str = "gelu",
):
    """Residual MLP — projection + n_blocks * (Linear→act→Linear→skip) + head."""
    _torch()
    import torch.nn as nn

    acts = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}
    if activation not in acts:
        raise ValueError(f"activation ∈ {list(acts)}")

    class _ResBlock(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
            self.act = acts[activation]()

        def forward(self, x):
            return x + self.fc2(self.act(self.fc1(x)))

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(int(in_dim), int(hidden))
            self.blocks = nn.Sequential(
                *map(lambda _: _ResBlock(int(hidden)), range(int(n_blocks)))
            )
            self.head = nn.Linear(int(hidden), int(out_dim))

        def forward(self, x):
            return self.head(self.blocks(self.proj(x)))

    return _Net()


def SirenMLP(
    in_dim: int, out_dim: int, hidden: int = 64, n_layers: int = 4,
    omega_0: float = 30.0,
):
    """SIREN — sin activation (고주파 / PINN 우수)."""
    torch = _torch()
    import torch.nn as nn

    class _Sine(nn.Module):
        def __init__(self, w0: float):
            super().__init__()
            self.w0 = float(w0)

        def forward(self, x):
            return torch.sin(self.w0 * x)

    layers: list = [nn.Linear(int(in_dim), int(hidden)), _Sine(omega_0)]
    layer_idx = 0
    while layer_idx < int(n_layers) - 2:
        layers.append(nn.Linear(int(hidden), int(hidden)))
        layers.append(_Sine(1.0))
        layer_idx += 1
    layers.append(nn.Linear(int(hidden), int(out_dim)))
    return nn.Sequential(*layers)


__all__ = ["MLP", "ResidualMLP", "SirenMLP"]
