"""Background job, cancellation, checkpoint, and resource tests."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from naviertwin.core.training import JobState, TrainingJobManager, training_preflight


def test_job_reports_progress_and_writes_checkpoint(tmp_path: Path) -> None:
    def task(context):
        context.report(0.25, "prepared")
        context.write_checkpoint(b"checkpoint")
        context.report(0.75, "trained")
        return 42

    with TrainingJobManager(max_workers=1, checkpoint_dir=tmp_path) as manager:
        job_id = manager.submit("rom", task, checkpoint_name="rom.ckpt")
        result = manager.wait(job_id, timeout=2)
        record = manager.record(job_id)

    assert result == 42
    assert record.state is JobState.COMPLETED
    assert record.progress == 1.0
    assert record.message == "trained"
    assert (tmp_path / "rom.ckpt").read_bytes() == b"checkpoint"


def test_running_job_can_be_cancelled() -> None:
    started = threading.Event()

    def task(context):
        started.set()
        while not context.cancelled:
            started.wait(0.005)
        context.raise_if_cancelled()

    with TrainingJobManager(max_workers=1) as manager:
        job_id = manager.submit("gino", task)
        assert started.wait(1)
        assert manager.cancel(job_id)
        assert manager.wait(job_id, timeout=2) is None
        record = manager.record(job_id)

    assert record.state is JobState.CANCELLED


def test_failed_job_retains_error() -> None:
    def task(_context):
        raise ValueError("bad training data")

    with TrainingJobManager(max_workers=1) as manager:
        job_id = manager.submit("operator", task)
        with pytest.raises(ValueError, match="bad training data"):
            manager.wait(job_id, timeout=2)
        record = manager.record(job_id)

    assert record.state is JobState.FAILED
    assert record.error == "ValueError: bad training data"


def test_cuda_preflight_recommends_reduction() -> None:
    result = training_preflight(
        required_bytes=10_000,
        device="cuda",
        reserve_fraction=0.2,
        available_cuda_bytes=5_000,
    )

    assert not result.ok
    assert result.usable_bytes == 4_000
    assert result.recommended_retain_fraction == pytest.approx(0.4)
    assert "40.0%" in result.reason


def test_auto_preflight_falls_back_to_cpu_without_cuda() -> None:
    result = training_preflight(
        required_bytes=100,
        device="auto",
        available_cuda_bytes=0,
    )

    assert result.ok
    assert result.device == "cpu"


def test_checkpoint_name_cannot_escape_checkpoint_directory(tmp_path: Path) -> None:
    with TrainingJobManager(checkpoint_dir=tmp_path) as manager:
        with pytest.raises(ValueError, match="inside checkpoint_dir"):
            manager.submit("rom", lambda _context: None, checkpoint_name="../escape.ckpt")


def test_shutdown_marks_queued_jobs_cancelled() -> None:
    release = threading.Event()

    def blocking(_context):
        release.wait(2)

    manager = TrainingJobManager(max_workers=1)
    running = manager.submit("rom", blocking)
    queued = manager.submit("rom", lambda _context: 1)
    for _ in range(100):
        if manager.record(running).state is JobState.RUNNING:
            break
        time.sleep(0.001)

    manager.shutdown(wait=False, cancel_pending=True)
    queued_record = manager.record(queued)
    release.set()

    assert queued_record.state is JobState.CANCELLED
