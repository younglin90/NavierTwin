"""학습 콜백 — EarlyStopping / LR scheduler wrapper / best-model tracker.

PyTorch 학습 루프에 바로 끼워 쓸 수 있는 경량 헬퍼.

Examples:
    >>> from naviertwin.utils.training_callbacks import EarlyStopping
    >>> es = EarlyStopping(patience=3, min_delta=1e-4)
    >>> es.step(1.0)
    False
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EarlyStopping:
    """loss 가 일정 기간 개선되지 않으면 stop 신호."""

    patience: int = 10
    min_delta: float = 1e-6
    mode: str = "min"  # "min" or "max"
    _best: float = field(default=float("inf"), init=False)
    _counter: int = field(default=0, init=False)
    stopped: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._best = float("inf") if self.mode == "min" else float("-inf")

    def _improved(self, v: float) -> bool:
        if self.mode == "min":
            return v < self._best - self.min_delta
        return v > self._best + self.min_delta

    def step(self, value: float) -> bool:
        """새 값 보고 → True 면 stop 권고."""
        if self._improved(value):
            self._best = float(value)
            self._counter = 0
        else:
            self._counter += 1
        if self._counter >= self.patience:
            self.stopped = True
        return self.stopped

    @property
    def best(self) -> float:
        return self._best


@dataclass
class BestModelTracker:
    """가장 좋은 모델의 state_dict 를 메모리에 보관."""

    mode: str = "min"
    _best_value: float = field(default=float("inf"), init=False)
    _best_state: Any = field(default=None, init=False)
    _best_epoch: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        self._best_value = float("inf") if self.mode == "min" else float("-inf")

    def _better(self, v: float) -> bool:
        return v < self._best_value if self.mode == "min" else v > self._best_value

    def step(self, model: Any, value: float, epoch: int) -> bool:
        """개선 시 state_dict 백업. 반환: 이번 step 에서 갱신됐는지."""
        if self._better(value):
            self._best_value = float(value)
            self._best_state = copy.deepcopy(model.state_dict())
            self._best_epoch = int(epoch)
            return True
        return False

    def restore(self, model: Any) -> Any:
        if self._best_state is None:
            raise RuntimeError("best 이력 없음")
        model.load_state_dict(self._best_state)
        logger.info(
            "best 모델 복원 (epoch=%d, value=%.6g)",
            self._best_epoch, self._best_value,
        )
        return model

    @property
    def best_value(self) -> float:
        return self._best_value

    @property
    def best_epoch(self) -> int:
        return self._best_epoch


def make_lr_scheduler(optimizer: Any, kind: str = "plateau", **kwargs: Any) -> Any:
    """torch.optim.lr_scheduler 팩토리 (plateau / cosine / step)."""
    try:
        from torch.optim import lr_scheduler
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc

    if kind == "plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "min"),
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 5),
        )
    if kind == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", 50),
            eta_min=kwargs.get("eta_min", 0.0),
        )
    if kind == "step":
        return lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 20),
            gamma=kwargs.get("gamma", 0.5),
        )
    raise ValueError(f"알 수 없는 scheduler: {kind}")


__all__ = ["EarlyStopping", "BestModelTracker", "make_lr_scheduler"]
