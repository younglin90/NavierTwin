"""Round 73 — PyTorch 모델 체크포인트 저장/복원."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestModelCheckpoint:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn

        from naviertwin.utils.model_checkpoint import (
            count_parameters,
            load_model,
            save_model,
        )

        torch.manual_seed(0)
        m1 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        opt = torch.optim.Adam(m1.parameters(), lr=1e-3)
        n_params = count_parameters(m1)
        assert n_params > 0

        out = tmp_path / "ckpt.pt"
        save_model(m1, out, optimizer=opt, meta={"epoch": 5, "loss": 0.01})
        assert out.exists()

        m2 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        ckpt = load_model(out, model=m2)
        assert ckpt["meta"]["epoch"] == 5
        assert ckpt["class_name"] == "Sequential"

        x = torch.randn(3, 4)
        with torch.no_grad():
            assert torch.allclose(m1(x), m2(x))

    def test_optimizer_restore(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn

        from naviertwin.utils.model_checkpoint import load_model, save_model

        m = nn.Linear(2, 1)
        opt = torch.optim.SGD(m.parameters(), lr=0.5, momentum=0.9)
        # step once to populate optimizer state
        x = torch.randn(4, 2)
        y = torch.randn(4, 1)
        loss = ((m(x) - y) ** 2).mean()
        loss.backward()
        opt.step()

        out = tmp_path / "ckpt.pt"
        save_model(m, out, optimizer=opt)

        m2 = nn.Linear(2, 1)
        opt2 = torch.optim.SGD(m2.parameters(), lr=0.5, momentum=0.9)
        load_model(out, model=m2, optimizer=opt2)
        assert opt2.param_groups[0]["lr"] == 0.5
        assert opt2.param_groups[0]["momentum"] == 0.9

    def test_missing_file_raises(self) -> None:
        pytest.importorskip("torch")
        from naviertwin.utils.model_checkpoint import load_model

        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent_ckpt.pt")
