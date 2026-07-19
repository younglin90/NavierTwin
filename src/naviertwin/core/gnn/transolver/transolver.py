"""Mesh-native Physics-Attention network inspired by Transolver.

The implementation keeps the useful Transolver invariant: attention cost is
linear in mesh point count and quadratic only in a small learned slice count.
It accepts variable-size point sets and does not require a common grid.
"""

from __future__ import annotations

from typing import Any


def _torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch 설치 필요: pip install naviertwin[ml]") from exc
    return torch, nn


class PhysicsAttentionBlock:
    """Factory-compatible wrapper for one learned physics-slice attention block."""

    def __new__(
        cls,
        hidden: int,
        n_slices: int = 32,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> Any:
        torch, nn = _torch_modules()
        if hidden <= 0 or n_slices <= 0 or n_heads <= 0:
            raise ValueError("hidden, n_slices, n_heads 는 양수여야 합니다.")
        if hidden % n_heads:
            raise ValueError("hidden 은 n_heads 로 나누어져야 합니다.")

        class _Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm1 = nn.LayerNorm(hidden)
                self.slice_proj = nn.Linear(hidden, n_slices)
                self.value_proj = nn.Linear(hidden, hidden)
                self.token_attention = nn.MultiheadAttention(
                    hidden, n_heads, dropout=dropout, batch_first=True
                )
                self.token_proj = nn.Linear(hidden, hidden)
                self.norm2 = nn.LayerNorm(hidden)
                self.mlp = nn.Sequential(
                    nn.Linear(hidden, hidden * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden * 2, hidden),
                )
                self.dropout = nn.Dropout(dropout)

            def forward(self, x: Any) -> Any:
                normalized = self.norm1(x)
                dispatch = torch.softmax(self.slice_proj(normalized), dim=-1)
                collect = dispatch / dispatch.sum(dim=0, keepdim=True).clamp_min(1e-6)
                tokens = collect.transpose(0, 1) @ self.value_proj(normalized)
                attended, _ = self.token_attention(
                    tokens.unsqueeze(0), tokens.unsqueeze(0), tokens.unsqueeze(0),
                    need_weights=False,
                )
                point_update = dispatch @ self.token_proj(attended.squeeze(0))
                x = x + self.dropout(point_update)
                return x + self.dropout(self.mlp(self.norm2(x)))

        return _Block()


class Transolver:
    """Factory for variable-size pointwise field regression with physics slices."""

    def __new__(
        cls,
        in_dim: int,
        out_dim: int,
        hidden: int = 64,
        n_layers: int = 4,
        n_slices: int = 32,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> Any:
        _torch, nn = _torch_modules()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim 과 out_dim 은 양수여야 합니다.")

        class _Transolver(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input = nn.Sequential(
                    nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
                )
                self.blocks = nn.ModuleList(
                    PhysicsAttentionBlock(hidden, n_slices, n_heads, dropout)
                    for _ in range(max(1, int(n_layers)))
                )
                self.output = nn.Sequential(
                    nn.LayerNorm(hidden), nn.Linear(hidden, out_dim)
                )

            def forward(self, x: Any) -> Any:
                state = self.input(x)
                for block in self.blocks:
                    state = block(state)
                return self.output(state)

        return _Transolver()


__all__ = ["PhysicsAttentionBlock", "Transolver"]
