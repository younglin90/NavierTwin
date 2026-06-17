"""DeepONet — branch+trunk.

u(y) ≈ Σ_k b_k(input_fn) · t_k(y).

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


def DeepONet(
    branch_input_dim: int,
    trunk_input_dim: int,
    *,
    p: int = 32,
    hidden: int = 64,
    n_layers: int = 3,
    activation: str = "relu",
):
    """
    Args:
        branch_input_dim: sensor values 수 (u 의 샘플링 크기).
        trunk_input_dim: 쿼리 좌표 차원.
        p: latent basis 크기.
    """
    _torch()
    import torch.nn as nn

    acts = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
    Act = acts.get(activation, nn.ReLU)

    def _mlp(in_d, out_d):
        layers = []
        prev = in_d
        layer = 0
        while layer < n_layers:
            layers.append(nn.Linear(prev, hidden))
            layers.append(Act())
            prev = hidden
            layer += 1
        layers.append(nn.Linear(prev, out_d))
        return nn.Sequential(*layers)

    import torch

    class _DON(nn.Module):
        def __init__(self):
            super().__init__()
            self.branch = _mlp(branch_input_dim, p)
            self.trunk = _mlp(trunk_input_dim, p)
            self.bias = nn.Parameter(torch.zeros(1))

        def forward(self, u, y):
            """u: (B, m), y: (B, q, d_y) 또는 (q, d_y).

            Returns:
                (B, q) scalar output.
            """
            b = self.branch(u)              # (B, p)
            if y.dim() == 2:
                t = self.trunk(y)           # (q, p)
                return torch.einsum("bp,qp->bq", b, t) + self.bias
            t = self.trunk(y)               # (B, q, p)
            return torch.einsum("bp,bqp->bq", b, t) + self.bias

    return _DON()


__all__ = ["DeepONet"]
