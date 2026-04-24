"""Round 148 — TorchScript export + verify."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestTorchScript:
    def test_trace_and_match(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn

        from naviertwin.core.export.torchscript_verify import (
            trace_and_save,
            verify_script_matches,
        )

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        x = torch.randn(3, 4)
        p = trace_and_save(model, x, tmp_path / "m.pt")
        assert p.exists()
        out = verify_script_matches(model, x, p)
        assert out["match"] is True
