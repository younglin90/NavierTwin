"""학습 콜백 시스템 — EarlyStopping / ProgressBar / LossLogger / Checkpoint.

사용 방법:
    for epoch in range(max_epochs):
        ...
        if not callback_manager.on_epoch_end(epoch, {"loss": loss_val}):
            break  # early stop

Examples:
    >>> from naviertwin.utils.callbacks import (
    ...     CallbackManager, EarlyStopping, LossLogger,
    ... )
    >>> m = CallbackManager([EarlyStopping(patience=3, min_delta=0.01)])
    >>> for ep in range(10):
    ...     cont = m.on_epoch_end(ep, {"loss": 1.0})  # plateau → 3 epoch 후 stop
    ...     if not cont:
    ...         break
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Callback:
    """콜백 베이스 — on_epoch_end 가 False 반환 시 학습 중단."""

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> bool:
        return True

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        pass


@dataclass
class EarlyStopping(Callback):
    """개선 없으면 조기 종료."""

    patience: int = 5
    min_delta: float = 0.0
    monitor: str = "loss"
    best: float = float("inf")
    n_bad: int = 0

    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> bool:
        v = float(logs.get(self.monitor, float("nan")))
        if v != v:  # NaN
            return True
        if v < self.best - self.min_delta:
            self.best = v
            self.n_bad = 0
        else:
            self.n_bad += 1
        if self.n_bad >= self.patience:
            logger.info(
                "EarlyStopping at epoch %d (no improvement for %d epochs, best=%.6g)",
                epoch, self.patience, self.best,
            )
            return False
        return True


@dataclass
class LossLogger(Callback):
    """epoch 별 logs 기록."""

    history: list[dict[str, Any]] = field(default_factory=list)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> bool:
        self.history.append({"epoch": epoch, **logs})
        return True


@dataclass
class ProgressBar(Callback):
    """tqdm 기반 진행도 (없으면 print 폴백)."""

    total: int = 100
    monitor: str = "loss"
    _bar: Any = None

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        try:
            from tqdm import tqdm

            self._bar = tqdm(total=self.total, desc="training")
        except ImportError:
            self._bar = None

    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> bool:
        v = logs.get(self.monitor)
        if self._bar is not None:
            self._bar.update(1)
            if v is not None:
                self._bar.set_postfix({self.monitor: f"{v:.4g}"})
        return True

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        if self._bar is not None:
            self._bar.close()


@dataclass
class ModelCheckpoint(Callback):
    """best monitor 갱신 시 save_fn 호출."""

    save_fn: Callable[[], None] = field(default=lambda: None)
    monitor: str = "loss"
    mode: str = "min"
    best: float = float("inf")

    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> bool:
        v = float(logs.get(self.monitor, float("nan")))
        if v != v:
            return True
        improved = (self.mode == "min" and v < self.best) or (
            self.mode == "max" and v > self.best
        )
        if improved:
            self.best = v
            try:
                self.save_fn()
                logger.info("Checkpoint saved at epoch %d (%s=%.6g)", epoch, self.monitor, v)
            except Exception as e:  # noqa: BLE001
                logger.warning("Checkpoint 저장 실패: %s", e)
        return True


@dataclass
class CallbackManager:
    """여러 콜백을 순차 호출."""

    callbacks: list[Callback] = field(default_factory=list)

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        for c in self.callbacks:
            c.on_train_begin(logs)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> bool:
        cont = True
        for c in self.callbacks:
            if not c.on_epoch_end(epoch, logs):
                cont = False
        return cont

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        for c in self.callbacks:
            c.on_train_end(logs)


def train_with_callbacks(
    loop: Callable[[int], dict[str, Any]],
    max_epochs: int,
    callbacks: list[Callback] | None = None,
) -> dict[str, Any]:
    """제공된 loop(epoch)→logs 를 콜백과 함께 실행.

    Returns:
        {"stopped_at": int, "last_logs": dict}.
    """
    mgr = CallbackManager(callbacks or [])
    mgr.on_train_begin({})
    last_logs: dict[str, Any] = {}
    stopped_at = max_epochs - 1
    for epoch in range(max_epochs):
        last_logs = loop(epoch)
        if not mgr.on_epoch_end(epoch, last_logs):
            stopped_at = epoch
            break
    mgr.on_train_end(last_logs)
    return {"stopped_at": stopped_at, "last_logs": last_logs}


__all__ = [
    "Callback",
    "EarlyStopping",
    "LossLogger",
    "ProgressBar",
    "ModelCheckpoint",
    "CallbackManager",
    "train_with_callbacks",
]
