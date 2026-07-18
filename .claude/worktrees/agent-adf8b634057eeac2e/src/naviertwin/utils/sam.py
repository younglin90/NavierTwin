"""Sharpness-Aware Minimization (SAM) — Foret et al. 2021.

Two-step: ascent ε on params, then gradient at perturbed point used to update.

Examples:
    >>> import pytest
    >>> torch = pytest.importorskip('torch')
    >>> from naviertwin.utils.sam import SAM
"""

from __future__ import annotations

from typing import Any


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


class SAM:
    """Wrap a torch optimizer to perform SAM two-step update."""

    def __init__(self, params: Any, base_optimizer: Any, *, rho: float = 0.05) -> None:
        if not has_torch():
            raise ImportError("torch not installed")
        self.params = list(params)
        self.base = base_optimizer
        self.rho = rho
        self._eps_state: list[Any] = []

    def first_step(self) -> None:
        import torch
        with torch.no_grad():
            grad_sum = 0.0
            idx = 0
            while idx < len(self.params):
                p = self.params[idx]
                if p.grad is not None:
                    grad_sum = grad_sum + p.grad.pow(2).sum()
                idx += 1
            grad_norm = torch.sqrt(grad_sum)
            scale = self.rho / (grad_norm + 1e-12)
            self._eps_state = []
            idx = 0
            while idx < len(self.params):
                p = self.params[idx]
                if p.grad is None:
                    self._eps_state.append(None)
                    idx += 1
                    continue
                e = p.grad * scale
                p.add_(e)
                self._eps_state.append(e)
                idx += 1

    def second_step(self) -> None:
        import torch
        with torch.no_grad():
            idx = 0
            if len(self.params) != len(self._eps_state):
                raise ValueError("SAM state length mismatch")
            while idx < len(self.params):
                p = self.params[idx]
                e = self._eps_state[idx]
                if e is not None:
                    p.sub_(e)
                idx += 1
        self.base.step()


__all__ = ["SAM", "has_torch"]
