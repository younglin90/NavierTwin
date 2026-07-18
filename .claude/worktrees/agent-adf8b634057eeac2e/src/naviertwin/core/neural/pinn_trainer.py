"""일반 PINN trainer — data loss + physics loss + boundary loss.

Examples:
    >>> import torch  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any, Callable


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc


class PINNTrainer:
    """Sum of weighted loss functions with gradient descent."""

    def __init__(
        self,
        model: Any,
        *,
        lr: float = 1e-3,
        weights: dict[str, float] | None = None,
        optimizer: str = "adam",
    ) -> None:
        torch = _torch()
        self.model = model
        self.weights = weights or {"data": 1.0, "physics": 1.0, "boundary": 1.0}
        if optimizer == "adam":
            self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer == "lbfgs":
            self.opt = torch.optim.LBFGS(model.parameters(), lr=lr)
        else:
            raise ValueError(f"unknown optimizer: {optimizer}")
        self.history: list[dict[str, float]] = []

    def train(
        self,
        loss_fns: dict[str, Callable[[Any], Any]],
        n_epochs: int = 1000,
        *, verbose: bool = False,
    ) -> list[dict[str, float]]:
        """loss_fns: {"data": fn, "physics": fn, "boundary": fn}. fn(model) → scalar tensor."""
        epoch = 0
        while epoch < n_epochs:
            def closure():
                self.opt.zero_grad()
                total = 0.0
                parts: dict[str, float] = {}

                def add_loss(item: tuple[str, Callable[[Any], Any]]) -> None:
                    nonlocal total
                    name, fn = item
                    w = self.weights.get(name, 1.0)
                    loss = fn(self.model)
                    total = total + w * loss
                    parts[name] = float(loss.detach())

                tuple(map(add_loss, loss_fns.items()))
                total.backward()
                self._parts = parts
                self._total = float(total.detach())
                return total

            self.opt.step(closure)
            self.history.append({"epoch": epoch, "total": self._total, **self._parts})
            if verbose and epoch % max(1, n_epochs // 10) == 0:
                import sys
                sys.stderr.write(f"[pinn] ep={epoch} total={self._total:.4g}\n")
            epoch += 1
        return self.history


__all__ = ["PINNTrainer"]
