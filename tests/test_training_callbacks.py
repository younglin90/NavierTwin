"""Round 77 — 학습 콜백 (EarlyStopping / BestModelTracker / LR scheduler)."""

from __future__ import annotations

import pytest


class TestEarlyStopping:
    def test_stops_after_patience(self) -> None:
        from naviertwin.utils.training_callbacks import EarlyStopping

        es = EarlyStopping(patience=2, min_delta=1e-4)
        # 개선 → 정체 → 정체 → stop
        assert es.step(1.0) is False
        assert es.step(0.9) is False  # 개선
        assert es.step(0.9) is False  # 정체 1
        assert es.step(0.9) is True   # 정체 2 → stop
        assert es.best == pytest.approx(0.9)

    def test_mode_max(self) -> None:
        from naviertwin.utils.training_callbacks import EarlyStopping

        es = EarlyStopping(patience=1, mode="max")
        es.step(0.5)
        es.step(0.7)  # 개선
        assert es.step(0.6) is True  # 악화 → stop


class TestBestModelTracker:
    def test_tracks_and_restores(self) -> None:
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn

        from naviertwin.utils.training_callbacks import BestModelTracker

        m = nn.Linear(2, 1)
        tracker = BestModelTracker(mode="min")
        tracker.step(m, 1.0, epoch=0)  # 저장

        # 가중치 변경
        with torch.no_grad():
            m.weight.fill_(99.0)
        tracker.step(m, 2.0, epoch=1)  # 악화 → 저장 안 됨
        assert tracker.best_epoch == 0

        tracker.restore(m)
        # 복원 후 weight 가 99 는 아니어야 함
        assert not torch.all(m.weight == 99.0)


class TestLRScheduler:
    def test_plateau_factory(self) -> None:
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn

        from naviertwin.utils.training_callbacks import make_lr_scheduler

        m = nn.Linear(2, 1)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        sch = make_lr_scheduler(opt, "plateau", patience=2)
        sch.step(1.0)  # 정상 호출
        assert sch is not None

    def test_cosine_and_step(self) -> None:
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn

        from naviertwin.utils.training_callbacks import make_lr_scheduler

        m = nn.Linear(2, 1)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        assert make_lr_scheduler(opt, "cosine", T_max=10) is not None
        assert make_lr_scheduler(opt, "step", step_size=5) is not None
