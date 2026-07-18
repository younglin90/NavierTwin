"""Round 68 — callbacks / early stopping / progress."""

from __future__ import annotations


class TestEarlyStopping:
    def test_stops_on_plateau(self) -> None:
        from naviertwin.utils.callbacks import EarlyStopping

        es = EarlyStopping(patience=3, min_delta=0.01, monitor="loss")
        # 초기 improvement 후 정체
        assert es.on_epoch_end(0, {"loss": 1.0}) is True
        assert es.on_epoch_end(1, {"loss": 0.9}) is True
        assert es.on_epoch_end(2, {"loss": 0.9}) is True  # plateau start
        assert es.on_epoch_end(3, {"loss": 0.9}) is True
        assert es.on_epoch_end(4, {"loss": 0.9}) is False  # stop

    def test_continuous_improvement_no_stop(self) -> None:
        from naviertwin.utils.callbacks import EarlyStopping

        es = EarlyStopping(patience=2, min_delta=0.0)
        for i, loss in enumerate([1.0, 0.9, 0.8, 0.7, 0.6]):
            assert es.on_epoch_end(i, {"loss": loss}) is True


class TestLossLogger:
    def test_accumulates(self) -> None:
        from naviertwin.utils.callbacks import LossLogger

        logger = LossLogger()
        logger.on_epoch_end(0, {"loss": 1.0})
        logger.on_epoch_end(1, {"loss": 0.5, "acc": 0.9})
        assert len(logger.history) == 2
        assert logger.history[1]["acc"] == 0.9


class TestModelCheckpoint:
    def test_save_on_improvement(self) -> None:
        from naviertwin.utils.callbacks import ModelCheckpoint

        calls = {"n": 0}

        def _save() -> None:
            calls["n"] += 1

        cb = ModelCheckpoint(save_fn=_save, monitor="loss", mode="min")
        cb.on_epoch_end(0, {"loss": 1.0})  # save
        cb.on_epoch_end(1, {"loss": 1.5})  # no save
        cb.on_epoch_end(2, {"loss": 0.7})  # save
        assert calls["n"] == 2


class TestCallbackManager:
    def test_multiple_callbacks(self) -> None:
        from naviertwin.utils.callbacks import (
            CallbackManager,
            EarlyStopping,
            LossLogger,
        )

        es = EarlyStopping(patience=2, min_delta=0.0)
        log = LossLogger()
        _ = CallbackManager([log, es])  # constructed for side-effect coverage

        def loop(ep: int) -> dict:
            return {"loss": 1.0}

        from naviertwin.utils.callbacks import train_with_callbacks

        res = train_with_callbacks(loop, max_epochs=10, callbacks=[log, es])
        # plateau → 조기 종료
        assert res["stopped_at"] < 9
        assert len(log.history) <= 10


class TestProgressBar:
    def test_no_tqdm_fallback(self) -> None:
        from naviertwin.utils.callbacks import ProgressBar

        pb = ProgressBar(total=3, monitor="loss")
        pb.on_train_begin()
        assert pb.on_epoch_end(0, {"loss": 1.0}) is True
        pb.on_train_end()
