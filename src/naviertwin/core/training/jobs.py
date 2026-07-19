"""Thread-safe background training jobs with cooperative cancellation."""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from naviertwin.utils.atomic_io import atomic_write_bytes


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobCancelled(RuntimeError):
    """Raised by a cooperative task after cancellation is requested."""


@dataclass
class JobRecord:
    """Serializable status snapshot for UI and experiment tracking."""

    job_id: str
    strategy: str
    state: JobState = JobState.QUEUED
    progress: float = 0.0
    message: str = ""
    created_at: str = field(default_factory=_utc_now)
    started_at: str = ""
    finished_at: str = ""
    checkpoint_path: str = ""
    error: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


class JobContext:
    """Task-side API for progress, cancellation, and atomic checkpoints."""

    def __init__(
        self,
        job_id: str,
        cancel_event: threading.Event,
        progress_callback: Callable[[float, str], None],
        checkpoint_path: Path | None,
    ) -> None:
        self.job_id = job_id
        self._cancel_event = cancel_event
        self._progress_callback = progress_callback
        self.checkpoint_path = checkpoint_path

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise JobCancelled(f"job cancelled: {self.job_id}")

    def report(self, progress: float, message: str = "") -> None:
        self.raise_if_cancelled()
        self._progress_callback(min(1.0, max(0.0, float(progress))), str(message))

    def write_checkpoint(self, payload: bytes) -> Path:
        self.raise_if_cancelled()
        if self.checkpoint_path is None:
            raise RuntimeError("this job has no checkpoint path")
        return atomic_write_bytes(self.checkpoint_path, payload)


Task = Callable[[JobContext], Any]


class TrainingJobManager:
    """Runs long model fits away from the UI thread."""

    def __init__(
        self,
        *,
        max_workers: int = 1,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="naviertwin-training"
        )
        self._checkpoint_dir = (
            None if checkpoint_dir is None else Path(checkpoint_dir).expanduser().resolve()
        )
        self._records: dict[str, JobRecord] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._closed = False

    def submit(
        self,
        strategy: str,
        task: Task,
        *,
        job_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        checkpoint_name: str | None = None,
    ) -> str:
        """Queue one task and return its stable job id."""

        if not strategy:
            raise ValueError("strategy must be non-empty")
        with self._lock:
            if self._closed:
                raise RuntimeError("training job manager is closed")
            resolved_id = job_id or uuid.uuid4().hex
            if resolved_id in self._records:
                raise ValueError(f"duplicate job id: {resolved_id}")
            checkpoint_path = None
            if checkpoint_name is not None:
                if self._checkpoint_dir is None:
                    raise ValueError("checkpoint_name requires checkpoint_dir")
                checkpoint_path = (self._checkpoint_dir / checkpoint_name).resolve()
                if not checkpoint_path.is_relative_to(self._checkpoint_dir):
                    raise ValueError("checkpoint_name must remain inside checkpoint_dir")
            record = JobRecord(
                job_id=resolved_id,
                strategy=strategy,
                checkpoint_path=str(checkpoint_path or ""),
                metadata=dict(metadata or {}),
            )
            cancel_event = threading.Event()
            self._records[resolved_id] = record
            self._cancel_events[resolved_id] = cancel_event
            future = self._executor.submit(
                self._run, resolved_id, task, cancel_event, checkpoint_path
            )
            self._futures[resolved_id] = future
            return resolved_id

    def _run(
        self,
        job_id: str,
        task: Task,
        cancel_event: threading.Event,
        checkpoint_path: Path | None,
    ) -> Any:
        with self._lock:
            record = self._records[job_id]
            if cancel_event.is_set():
                record.state = JobState.CANCELLED
                record.finished_at = _utc_now()
                return None
            record.state = JobState.RUNNING
            record.started_at = _utc_now()

        def update(progress: float, message: str) -> None:
            with self._lock:
                current = self._records[job_id]
                current.progress = progress
                current.message = message

        context = JobContext(job_id, cancel_event, update, checkpoint_path)
        try:
            result = task(context)
            context.raise_if_cancelled()
        except JobCancelled:
            with self._lock:
                record = self._records[job_id]
                record.state = JobState.CANCELLED
                record.finished_at = _utc_now()
                record.message = "cancelled"
            return None
        except Exception as exc:
            with self._lock:
                record = self._records[job_id]
                record.state = JobState.FAILED
                record.finished_at = _utc_now()
                record.error = f"{type(exc).__name__}: {exc}"
            raise
        with self._lock:
            record = self._records[job_id]
            record.state = JobState.COMPLETED
            record.progress = 1.0
            record.finished_at = _utc_now()
            if not record.message:
                record.message = "completed"
        return result

    def cancel(self, job_id: str) -> bool:
        """Request cooperative cancellation; queued tasks cancel immediately."""

        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                raise KeyError(f"unknown job: {job_id}")
            if record.state in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}:
                return False
            self._cancel_events[job_id].set()
            future = self._futures[job_id]
            if future.cancel():
                record.state = JobState.CANCELLED
                record.finished_at = _utc_now()
                record.message = "cancelled"
            return True

    def record(self, job_id: str) -> JobRecord:
        """Return a detached status snapshot."""

        with self._lock:
            try:
                current = self._records[job_id]
            except KeyError as exc:
                raise KeyError(f"unknown job: {job_id}") from exc
            return replace(current, metadata=dict(current.metadata))

    def list_records(self) -> list[JobRecord]:
        with self._lock:
            return [self.record(job_id) for job_id in self._records]

    def wait(self, job_id: str, timeout: float | None = None) -> Any:
        """Wait for one task and return its result."""

        with self._lock:
            try:
                future = self._futures[job_id]
            except KeyError as exc:
                raise KeyError(f"unknown job: {job_id}") from exc
        return future.result(timeout=timeout)

    def shutdown(self, *, wait: bool = True, cancel_pending: bool = False) -> None:
        """Close the manager; optionally cancel unfinished jobs."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            if cancel_pending:
                for job_id, record in self._records.items():
                    if record.state in {JobState.QUEUED, JobState.RUNNING}:
                        self._cancel_events[job_id].set()
                    if record.state is JobState.QUEUED:
                        record.state = JobState.CANCELLED
                        record.finished_at = _utc_now()
                        record.message = "cancelled"
        self._executor.shutdown(wait=wait, cancel_futures=cancel_pending)

    def __enter__(self) -> TrainingJobManager:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.shutdown()


__all__ = [
    "JobCancelled",
    "JobContext",
    "JobRecord",
    "JobState",
    "TrainingJobManager",
]
