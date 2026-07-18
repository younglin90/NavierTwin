"""Logger configuration regression tests."""

from __future__ import annotations

import logging
from pathlib import Path


def _reset_navier_logger() -> None:
    import naviertwin.utils.logger as logger_mod

    root = logging.getLogger("naviertwin")
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    logger_mod._root_configured = False


def test_logger_respects_env_log_dir(tmp_path: Path, monkeypatch) -> None:
    """NAVIER_TWIN_LOG_DIR should redirect the rotating file handler."""
    import naviertwin.utils.logger as logger_mod

    _reset_navier_logger()
    monkeypatch.setenv("NAVIER_TWIN_LOG_DIR", str(tmp_path))

    log = logger_mod.get_logger("naviertwin.test_logger")
    log.debug("env log dir smoke")
    for handler in logging.getLogger("naviertwin").handlers:
        handler.flush()

    assert (tmp_path / "naviertwin.log").exists()
    _reset_navier_logger()


def test_logger_falls_back_to_temp_when_default_unwritable(monkeypatch) -> None:
    """A read-only default log directory should not emit a warning handler path."""
    import naviertwin.utils.logger as logger_mod

    _reset_navier_logger()
    monkeypatch.delenv("NAVIER_TWIN_LOG_DIR", raising=False)
    monkeypatch.setattr(logger_mod, "_DEFAULT_LOG_DIR", Path("/sys/naviertwin/logs"))

    log = logger_mod.get_logger("naviertwin.test_logger")
    log.debug("fallback smoke")

    handlers = logging.getLogger("naviertwin").handlers
    assert any(isinstance(handler, logging.handlers.RotatingFileHandler) for handler in handlers)
    _reset_navier_logger()
