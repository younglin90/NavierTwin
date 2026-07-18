"""Round 411 — gradient checkpoint."""

from __future__ import annotations

import pytest


class TestGradCkpt:
    def test_seq_runs(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.utils.grad_checkpoint import checkpoint_seq

        m1 = torch.nn.Linear(4, 4)
        m2 = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4, requires_grad=True)
        y = checkpoint_seq([m1, m2], x)
        y.sum().backward()
        assert x.grad is not None
